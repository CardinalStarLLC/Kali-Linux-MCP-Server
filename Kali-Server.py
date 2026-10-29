#!/usr/bin/env python3

# This script connect the MCP AI agent to Kali Linux terminal and API Server.

# some of the code here was inspired from https://github.com/whit3rabbit0/project_astro , be sure to check them out

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_PORT = int(os.environ.get("API_PORT", 5000))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0").lower() in ("1", "true", "yes", "y")
COMMAND_TIMEOUT = 600       # 10 minutes — worker-side hard limit
JOB_RETENTION_SECONDS = 3600  # Completed jobs are kept for 1 hour then pruned

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------

class JobStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class JobStore:
    """Thread-safe in-memory store for async job state and results."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, description: str) -> Dict[str, Any]:
        job = {
            "job_id":      job_id,
            "description": description,
            "status":      JobStatus.PENDING,
            "created_at":  time.time(),
            "started_at":  None,
            "finished_at": None,
            "result":      None,
            "error":       None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]     = JobStatus.RUNNING
                self._jobs[job_id]["started_at"] = time.time()

    def set_completed(self, job_id: str, result: Dict[str, Any]):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]      = JobStatus.COMPLETED
                self._jobs[job_id]["finished_at"] = time.time()
                self._jobs[job_id]["result"]      = result

    def set_failed(self, job_id: str, error: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]      = JobStatus.FAILED
                self._jobs[job_id]["finished_at"] = time.time()
                self._jobs[job_id]["error"]       = error

    def list_all(self) -> list:
        with self._lock:
            return list(self._jobs.values())

    def prune_old(self):
        """Remove completed/failed jobs older than JOB_RETENTION_SECONDS."""
        cutoff = time.time() - JOB_RETENTION_SECONDS
        with self._lock:
            stale = [
                jid for jid, job in self._jobs.items()
                if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED)
                and job.get("finished_at", time.time()) < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]
        if stale:
            logger.info(f"Pruned {len(stale)} stale jobs")


job_store = JobStore()
_job_queue: "queue.Queue[tuple]" = queue.Queue()  # (job_id, callable)


def _worker_loop():
    """Background thread: pull jobs from the queue and execute them."""
    logger.info("Job worker thread started")
    while True:
        try:
            job_id, fn = _job_queue.get(timeout=5)
        except queue.Empty:
            job_store.prune_old()
            continue

        job_store.set_running(job_id)
        logger.info(f"[{job_id}] Starting job")
        try:
            result = fn()
            job_store.set_completed(job_id, result)
            logger.info(f"[{job_id}] Job completed (success={result.get('success')})")
        except Exception as exc:
            logger.error(f"[{job_id}] Job raised exception: {exc}")
            logger.error(traceback.format_exc())
            job_store.set_failed(job_id, str(exc))
        finally:
            _job_queue.task_done()


_worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="job-worker")
_worker_thread.start()


def enqueue(description: str, fn: Callable) -> Dict[str, Any]:
    """
    Enqueue a callable for async execution.

    Returns the initial job record (status=pending) immediately so the
    caller can respond with a 202 and a job_id.
    """
    job_id = str(uuid.uuid4())
    job = job_store.create(job_id, description)
    _job_queue.put((job_id, fn))
    logger.info(f"[{job_id}] Enqueued: {description}")
    return job


# ---------------------------------------------------------------------------
# Command Executor (unchanged logic, extracted to reusable helper)
# ---------------------------------------------------------------------------

class CommandExecutor:
    """Handles command execution with graceful timeout management."""

    def __init__(self, command: str, timeout: int = COMMAND_TIMEOUT):
        self.command = command
        self.timeout = timeout
        self.process = None
        self.stdout_data = ""
        self.stderr_data = ""
        self.return_code = None
        self.timed_out = False

    def _read_stdout(self):
        for line in iter(self.process.stdout.readline, ""):
            self.stdout_data += line

    def _read_stderr(self):
        for line in iter(self.process.stderr.readline, ""):
            self.stderr_data += line

    def execute(self) -> Dict[str, Any]:
        logger.info(f"Executing command: {self.command}")
        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            t_out = threading.Thread(target=self._read_stdout, daemon=True)
            t_err = threading.Thread(target=self._read_stderr, daemon=True)
            t_out.start()
            t_err.start()

            try:
                self.return_code = self.process.wait(timeout=self.timeout)
                t_out.join()
                t_err.join()
            except subprocess.TimeoutExpired:
                self.timed_out = True
                logger.warning(f"Command timed out after {self.timeout}s. Terminating.")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Process not responding to SIGTERM — killing.")
                    self.process.kill()
                self.return_code = -1

            success = (
                True if self.timed_out and (self.stdout_data or self.stderr_data)
                else (self.return_code == 0)
            )
            return {
                "stdout":          self.stdout_data,
                "stderr":          self.stderr_data,
                "return_code":     self.return_code,
                "success":         success,
                "timed_out":       self.timed_out,
                "partial_results": self.timed_out and bool(self.stdout_data or self.stderr_data),
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            logger.error(traceback.format_exc())
            return {
                "stdout":          self.stdout_data,
                "stderr":          f"Error: {e}\n{self.stderr_data}",
                "return_code":     -1,
                "success":         False,
                "timed_out":       False,
                "partial_results": bool(self.stdout_data or self.stderr_data),
            }


def run_command(command: str) -> Dict[str, Any]:
    return CommandExecutor(command).execute()


# ---------------------------------------------------------------------------
# Helper: build a quick 202 response
# ---------------------------------------------------------------------------

def accepted(job: Dict[str, Any]):
    """Return HTTP 202 Accepted with job metadata."""
    return jsonify({
        "job_id":      job["job_id"],
        "status":      job["status"],
        "description": job["description"],
    }), 202


# ---------------------------------------------------------------------------
# Job status / management endpoints
# ---------------------------------------------------------------------------

@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    """Poll the status (and result) of an async job."""
    job = job_store.get(job_id)
    if job is None:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404
    return jsonify(job)


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """List all tracked jobs (useful for debugging)."""
    return jsonify(job_store.list_all())


# ---------------------------------------------------------------------------
# Tool endpoints — all return 202 + job_id immediately
# ---------------------------------------------------------------------------

@app.route("/api/command", methods=["POST"])
def generic_command():
    """Enqueue an arbitrary shell command."""
    try:
        params = request.json or {}
        command = params.get("command", "")
        if not command:
            return jsonify({"error": "Command parameter is required"}), 400

        job = enqueue(f"command: {command[:80]}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/nmap", methods=["POST"])
def nmap():
    """Enqueue an nmap scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        scan_type       = params.get("scan_type", "-sCV")
        ports           = params.get("ports", "")
        additional_args = params.get("additional_args", "-T4 -Pn")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"nmap {scan_type}"
        if ports:
            command += f" -p {ports}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {target}"

        job = enqueue(f"nmap {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/gobuster", methods=["POST"])
def gobuster():
    """Enqueue a gobuster scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        mode            = params.get("mode", "dir")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400
        if mode not in ("dir", "dns", "fuzz", "vhost"):
            return jsonify({"error": f"Invalid mode: {mode}"}), 400

        command = f"gobuster {mode} -u {url} -w {wordlist}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"gobuster {mode} {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/dirb", methods=["POST"])
def dirb():
    """Enqueue a dirb scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"dirb {url} {wordlist}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"dirb {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/nikto", methods=["POST"])
def nikto():
    """Enqueue a nikto scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"nikto -h {target}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"nikto {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/sqlmap", methods=["POST"])
def sqlmap():
    """Enqueue a sqlmap scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        post_data       = params.get("data", "")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"sqlmap -u {url} --batch"
        if post_data:
            command += f" --data={post_data}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"sqlmap {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/metasploit", methods=["POST"])
def metasploit():
    """Enqueue a Metasploit module run."""
    try:
        params = request.json or {}
        module  = params.get("module", "")
        options = params.get("options", {})

        if not module:
            return jsonify({"error": "Module parameter is required"}), 400

        resource_file = f"/tmp/msf_{uuid.uuid4().hex}.rc"
        resource_lines = [f"use {module}"]
        for key, value in options.items():
            resource_lines.append(f"set {key} {value}")
        resource_lines.append("run")
        resource_content = "\n".join(resource_lines) + "\n"

        with open(resource_file, "w") as f:
            f.write(resource_content)

        def _run(cmd=f"msfconsole -q -r {resource_file}", rc=resource_file):
            result = run_command(cmd)
            try:
                os.remove(rc)
            except Exception:
                pass
            return result

        job = enqueue(f"metasploit {module}", _run)
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/hydra", methods=["POST"])
def hydra():
    """Enqueue a Hydra password attack."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        service         = params.get("service", "")
        username        = params.get("username", "")
        username_file   = params.get("username_file", "")
        password        = params.get("password", "")
        password_file   = params.get("password_file", "")
        additional_args = params.get("additional_args", "")

        if not target or not service:
            return jsonify({"error": "Target and service parameters are required"}), 400
        if not (username or username_file) or not (password or password_file):
            return jsonify({"error": "Username/username_file and password/password_file are required"}), 400

        command = "hydra -t 4"
        command += f" -l {username}" if username else f" -L {username_file}"
        command += f" -p {password}" if password else f" -P {password_file}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {target} {service}"

        job = enqueue(f"hydra {service}@{target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/john", methods=["POST"])
def john():
    """Enqueue a John the Ripper cracking job."""
    try:
        params = request.json or {}
        hash_file       = params.get("hash_file", "")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/rockyou.txt")
        format_type     = params.get("format", "")
        additional_args = params.get("additional_args", "")

        if not hash_file:
            return jsonify({"error": "Hash file parameter is required"}), 400

        command = "john"
        if format_type:
            command += f" --format={format_type}"
        if wordlist:
            command += f" --wordlist={wordlist}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {hash_file}"

        job = enqueue(f"john {hash_file}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/wpscan", methods=["POST"])
def wpscan():
    """Enqueue a WPScan scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"wpscan --url {url}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"wpscan {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/enum4linux", methods=["POST"])
def enum4linux():
    """Enqueue an enum4linux scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        additional_args = params.get("additional_args", "-a")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"enum4linux {additional_args} {target}"

        job = enqueue(f"enum4linux {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Red Team Tool Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/tools/theharvester", methods=["POST"])
def theharvester():
    """
    Enqueue a theHarvester OSINT scan.

    Tier 1 — Low risk. Passive OSINT only.
    Harvests emails, subdomains, hosts, and employee names from public sources.

    Required params:
        domain          Target domain to harvest

    Optional params:
        sources         Comma-separated data sources (default: broad passive set)
        limit           Max results to retrieve (default: 500)
        dns_resolve     Resolve discovered hosts, bool (default: false)
        additional_args Extra theHarvester arguments
    """
    try:
        params = request.json or {}
        domain          = params.get("domain", "")
        sources         = params.get("sources", "baidu,bing,crtsh,dnsdumpster,duckduckgo,hackertarget,rapiddns,urlscan")
        limit           = int(params.get("limit", 500))
        dns_resolve     = params.get("dns_resolve", False)
        additional_args = params.get("additional_args", "")

        if not domain:
            return jsonify({"error": "domain parameter is required"}), 400

        command = f"theHarvester -d {domain} -b {sources} -l {limit}"
        if dns_resolve:
            command += " -r"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"theharvester {domain}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/amass", methods=["POST"])
def amass():
    """
    Enqueue an Amass subdomain enumeration.

    Tier 1 (passive) / Tier 2 (active/brute).
    Deep subdomain enumeration using passive and active techniques.

    Required params:
        domain          Target domain

    Optional params:
        mode            passive | active | brute (default: passive)
        wordlist        Path to wordlist — used when mode is 'brute'
        output_dir      Directory for Amass output files (default: /tmp/amass_out)
        additional_args Extra Amass arguments
    """
    try:
        params = request.json or {}
        domain          = params.get("domain", "")
        mode            = params.get("mode", "passive")
        wordlist        = params.get("wordlist", "")
        output_dir      = params.get("output_dir", "/tmp/amass_out")
        additional_args = params.get("additional_args", "")

        if not domain:
            return jsonify({"error": "domain parameter is required"}), 400
        if mode not in ("passive", "active", "brute"):
            return jsonify({"error": f"Invalid mode '{mode}'. Must be: passive, active, brute"}), 400
        if mode == "brute" and not wordlist:
            return jsonify({"error": "wordlist is required when mode is 'brute'"}), 400

        command = f"amass enum -{mode} -d {domain} -dir {output_dir}"
        if mode == "brute" and wordlist:
            command += f" -brute -w {wordlist}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"amass {mode} {domain}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/recon-ng", methods=["POST"])
def recon_ng():
    """
    Enqueue a Recon-ng module run.

    Tier 1 — Low risk. Modular OSINT framework.

    Required params:
        module          Full module path e.g. recon/domains-hosts/hackertarget
        source          Source value (domain, IP, company name, etc.)

    Optional params:
        workspace       Recon-ng workspace name (default: default)
        additional_args Extra key=value options passed to the module
    """
    try:
        params = request.json or {}
        module          = params.get("module", "")
        source          = params.get("source", "")
        workspace       = params.get("workspace", "default")
        additional_args = params.get("additional_args", "")

        if not module:
            return jsonify({"error": "module parameter is required"}), 400
        if not source:
            return jsonify({"error": "source parameter is required"}), 400

        # Build a recon-ng script file for non-interactive execution
        script_file = f"/tmp/recon_ng_{uuid.uuid4().hex}.rc"
        script_lines = [
            f"workspaces select {workspace}",
            f"modules load {module}",
            f"options set SOURCE {source}",
        ]
        if additional_args:
            for opt in additional_args.split(","):
                opt = opt.strip()
                if "=" in opt:
                    k, v = opt.split("=", 1)
                    script_lines.append(f"options set {k.strip()} {v.strip()}")
        script_lines.append("run")
        script_lines.append("exit")

        with open(script_file, "w") as f:
            f.write("\n".join(script_lines) + "\n")

        def _run(cmd=f"recon-ng -r {script_file}", rc=script_file):
            result = run_command(cmd)
            try:
                os.remove(rc)
            except Exception:
                pass
            return result

        job = enqueue(f"recon-ng {module} ({source})", _run)
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/netexec", methods=["POST"])
def netexec():
    """
    Enqueue a NetExec (nxc) lateral movement/enumeration run.

    Tier 2–3. SMB, WinRM, LDAP, RDP, SSH testing across Windows environments.

    Required params:
        target          IP, CIDR, or hostname
        protocol        smb | winrm | ldap | rdp | ssh | mssql

    Optional params:
        username        Single username
        username_file   Path to username list
        password        Single password
        password_file   Path to password list
        hash            NTLM hash for Pass-the-Hash (format: LMHASH:NTHASH or just NTHASH)
        action          Module/action flag e.g. --shares, --users, --sam, --lsa
        domain          Windows domain name
        additional_args Extra netexec arguments
    """
    try:
        params = request.json or {}
        target          = params.get("target", "")
        protocol        = params.get("protocol", "")
        username        = params.get("username", "")
        username_file   = params.get("username_file", "")
        password        = params.get("password", "")
        password_file   = params.get("password_file", "")
        ntlm_hash       = params.get("hash", "")
        action          = params.get("action", "")
        domain          = params.get("domain", "")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "target parameter is required"}), 400
        if not protocol:
            return jsonify({"error": "protocol parameter is required"}), 400

        valid_protocols = ("smb", "winrm", "ldap", "rdp", "ssh", "mssql")
        if protocol not in valid_protocols:
            return jsonify({"error": f"Invalid protocol '{protocol}'. Must be one of: {', '.join(valid_protocols)}"}), 400

        command = f"nxc {protocol} {target}"
        if domain:
            command += f" -d {domain}"
        if username:
            command += f" -u {username}"
        elif username_file:
            command += f" -U {username_file}"
        if ntlm_hash:
            command += f" -H {ntlm_hash}"
        elif password:
            command += f" -p {password}"
        elif password_file:
            command += f" -P {password_file}"
        if action:
            command += f" {action}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"netexec {protocol} {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/impacket", methods=["POST"])
def impacket():
    """
    Enqueue an Impacket script execution.

    Tier 3 — High risk. Windows/AD protocol abuse.
    Use only on explicitly authorized targets.

    Required params:
        script          Script name: secretsdump | GetNPUsers | GetUserSPNs |
                        psexec | wmiexec | smbclient | ticketer | getPac
        target          Target IP or hostname

    Optional params:
        username        Username
        password        Password
        hash            NTLM hash (format: LMHASH:NTHASH or just NTHASH)
        domain          Domain name
        dc_ip           Domain controller IP
        additional_args Extra arguments for the specific script
    """
    try:
        params = request.json or {}
        script          = params.get("script", "")
        target          = params.get("target", "")
        username        = params.get("username", "")
        password        = params.get("password", "")
        ntlm_hash       = params.get("hash", "")
        domain          = params.get("domain", "")
        dc_ip           = params.get("dc_ip", "")
        additional_args = params.get("additional_args", "")

        valid_scripts = (
            "secretsdump", "GetNPUsers", "GetUserSPNs",
            "psexec", "wmiexec", "smbclient", "ticketer", "getPac",
        )
        if not script:
            return jsonify({"error": "script parameter is required"}), 400
        if script not in valid_scripts:
            return jsonify({"error": f"Invalid script '{script}'. Must be one of: {', '.join(valid_scripts)}"}), 400
        if not target:
            return jsonify({"error": "target parameter is required"}), 400

        # Build auth string: DOMAIN/user:pass@target  or  DOMAIN/user@target (hash separately)
        auth_user = f"{domain}\\{username}" if domain and username else username
        if password and not ntlm_hash:
            auth_string = f"{auth_user}:{password}@{target}" if auth_user else f"@{target}"
        else:
            auth_string = f"{auth_user}@{target}" if auth_user else f"@{target}"

        command = f"impacket-{script} {auth_string}"
        if ntlm_hash:
            # Impacket expects LMHASH:NTHASH — prepend empty LM if only NT provided
            hash_val = ntlm_hash if ":" in ntlm_hash else f":{ntlm_hash}"
            command += f" -hashes {hash_val}"
        if dc_ip:
            command += f" -dc-ip {dc_ip}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"impacket-{script} {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/responder", methods=["POST"])
def responder():
    """
    Enqueue a Responder LLMNR/NBT-NS/mDNS poisoning run.

    Tier 3 — High risk. Captures NTLMv2 hashes on internal segments.
    Use only on explicitly authorized internal network segments.

    Required params:
        interface       Network interface to listen on (e.g. eth0)

    Optional params:
        duration_seconds  How long to run before auto-stopping (default: 60)
        modes           Extra mode flags e.g. --wpad --lm (default: --wpad)
        additional_args Extra Responder arguments
    """
    try:
        params = request.json or {}
        interface        = params.get("interface", "")
        duration_seconds = int(params.get("duration_seconds", 60))
        modes            = params.get("modes", "--wpad")
        additional_args  = params.get("additional_args", "")

        if not interface:
            return jsonify({"error": "interface parameter is required"}), 400
        if duration_seconds < 1 or duration_seconds > 3600:
            return jsonify({"error": "duration_seconds must be between 1 and 3600"}), 400

        command = f"timeout {duration_seconds} responder -I {interface} {modes}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"responder {interface}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/evil-winrm", methods=["POST"])
def evil_winrm():
    """
    Enqueue an Evil-WinRM command execution.

    Tier 3 — High risk. WinRM post-exploitation shell access.
    Runs a single non-interactive command rather than an interactive shell.

    Required params:
        target          Target IP or hostname
        username        Username
        command         Command to execute on the remote host

    Optional params:
        password        Password (use either password or hash, not both)
        hash            NTLM hash for Pass-the-Hash
        additional_args Extra Evil-WinRM arguments
    """
    try:
        params = request.json or {}
        target          = params.get("target", "")
        username        = params.get("username", "")
        password        = params.get("password", "")
        ntlm_hash       = params.get("hash", "")
        command_to_run  = params.get("command", "")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "target parameter is required"}), 400
        if not username:
            return jsonify({"error": "username parameter is required"}), 400
        if not command_to_run:
            return jsonify({"error": "command parameter is required"}), 400
        if not password and not ntlm_hash:
            return jsonify({"error": "Either password or hash is required"}), 400

        command = f"evil-winrm -i {target} -u {username}"
        if ntlm_hash:
            command += f" -H {ntlm_hash}"
        elif password:
            command += f" -p {password}"
        # Execute a single command via -e flag (non-interactive)
        command += f" -e \"{command_to_run}\""
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"evil-winrm {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/bloodhound", methods=["POST"])
def bloodhound():
    """
    Enqueue a BloodHound data collection run (bloodhound-python).

    Tier 2 — Medium risk. Collects AD data for BloodHound attack path analysis.
    Collection only — visualization requires the BloodHound UI separately.

    Required params:
        domain          Target domain
        dc_ip           Domain controller IP
        username        Domain username

    Optional params:
        password            Password
        hash                NTLM hash (format: LMHASH:NTHASH or just NTHASH)
        collection_methods  Default | All | DCOnly | Session | Trusts | ACL | Group
                            (default: Default)
        output_dir          Directory for JSON output files (default: /tmp/bh_out)
        additional_args     Extra bloodhound-python arguments
    """
    try:
        params = request.json or {}
        domain              = params.get("domain", "")
        dc_ip               = params.get("dc_ip", "")
        username            = params.get("username", "")
        password            = params.get("password", "")
        ntlm_hash           = params.get("hash", "")
        collection_methods  = params.get("collection_methods", "Default")
        output_dir          = params.get("output_dir", "/tmp/bh_out")
        additional_args     = params.get("additional_args", "")

        if not domain:
            return jsonify({"error": "domain parameter is required"}), 400
        if not dc_ip:
            return jsonify({"error": "dc_ip parameter is required"}), 400
        if not username:
            return jsonify({"error": "username parameter is required"}), 400
        if not password and not ntlm_hash:
            return jsonify({"error": "Either password or hash is required"}), 400

        valid_methods = ("Default", "All", "DCOnly", "Session", "Trusts", "ACL", "Group")
        if collection_methods not in valid_methods:
            return jsonify({"error": f"Invalid collection_methods. Must be one of: {', '.join(valid_methods)}"}), 400

        os.makedirs(output_dir, exist_ok=True)

        command = (
            f"bloodhound-python -d {domain} -u {username} -ns {dc_ip}"
            f" -c {collection_methods} -o {output_dir} --zip"
        )
        if ntlm_hash:
            hash_val = ntlm_hash if ":" in ntlm_hash else f":{ntlm_hash}"
            command += f" --hashes {hash_val}"
        elif password:
            command += f" -p {password}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"bloodhound-python {domain}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/aircrack", methods=["POST"])
def aircrack():
    """
    Enqueue an aircrack-ng Wi-Fi action.

    Tier 3 — High risk. Wi-Fi assessment (monitor mode, capture, deauth, crack).
    Use only on networks you are explicitly authorized to test.

    Required params:
        action          monitor_start | capture | deauth | crack
        interface       Wireless interface (e.g. wlan0)

    Optional params (by action):
        bssid           Target AP BSSID — required for capture/deauth
        channel         Channel to lock on — used for capture (default: 0 = auto)
        capture_file    Output/input .cap file path (default: /tmp/capture)
        wordlist        Wordlist path — used for crack (default: rockyou.txt)
        duration_seconds  Capture run duration in seconds (default: 60)
        additional_args Extra arguments
    """
    try:
        params = request.json or {}
        action           = params.get("action", "")
        interface        = params.get("interface", "")
        bssid            = params.get("bssid", "")
        channel          = int(params.get("channel", 0))
        capture_file     = params.get("capture_file", "/tmp/capture")
        wordlist         = params.get("wordlist", "/usr/share/wordlists/rockyou.txt")
        duration_seconds = int(params.get("duration_seconds", 60))
        additional_args  = params.get("additional_args", "")

        valid_actions = ("monitor_start", "capture", "deauth", "crack")
        if not action:
            return jsonify({"error": "action parameter is required"}), 400
        if action not in valid_actions:
            return jsonify({"error": f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"}), 400
        if not interface:
            return jsonify({"error": "interface parameter is required"}), 400

        if action == "monitor_start":
            command = f"airmon-ng start {interface}"
        elif action == "capture":
            if not bssid:
                return jsonify({"error": "bssid is required for capture action"}), 400
            command = f"timeout {duration_seconds} airodump-ng"
            command += f" --bssid {bssid}"
            if channel:
                command += f" --channel {channel}"
            command += f" -w {capture_file} {interface}mon"
        elif action == "deauth":
            if not bssid:
                return jsonify({"error": "bssid is required for deauth action"}), 400
            command = f"aireplay-ng --deauth 10 -a {bssid} {interface}mon"
        elif action == "crack":
            command = f"aircrack-ng {capture_file}*.cap -w {wordlist}"

        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"aircrack {action} {interface}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/bettercap", methods=["POST"])
def bettercap():
    """
    Enqueue a Bettercap network attack/recon run.

    Tier 3 — High risk. Network recon, MITM, ARP spoofing, Wi-Fi, and BLE.
    Use only on explicitly authorized network segments.

    Required params:
        interface       Network interface (e.g. eth0)

    Optional params:
        caplet          Path to a .cap caplet script file
        commands        Semicolon-separated bettercap commands to evaluate
        duration_seconds  Run duration before auto-stop (default: 60)
        additional_args Extra bettercap arguments
    """
    try:
        params = request.json or {}
        interface        = params.get("interface", "")
        caplet           = params.get("caplet", "")
        commands         = params.get("commands", "")
        duration_seconds = int(params.get("duration_seconds", 60))
        additional_args  = params.get("additional_args", "")

        if not interface:
            return jsonify({"error": "interface parameter is required"}), 400
        if not caplet and not commands:
            return jsonify({"error": "Either caplet or commands is required"}), 400
        if duration_seconds < 1 or duration_seconds > 3600:
            return jsonify({"error": "duration_seconds must be between 1 and 3600"}), 400

        command = f"timeout {duration_seconds} bettercap -iface {interface}"
        if caplet:
            command += f" -caplet {caplet}"
        if commands:
            command += f" -eval \"{commands}\""
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"bettercap {interface}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Blue Team Tool Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/tools/volatility", methods=["POST"])
def volatility():
    """
    Enqueue a Volatility 3 memory forensics run.

    Tier 1 — Low risk. Forensic analysis of memory dumps.
    No active network testing — safe for incident response workflows.

    Required params:
        image           Path to memory dump file
        plugin          Plugin to run e.g. windows.pslist, windows.netscan,
                        windows.malfind, windows.hashdump, windows.cmdline,
                        windows.dlllist, windows.dumpfiles, linux.pslist,
                        linux.bash

    Optional params:
        pid             Filter by PID (0 = all, default: 0)
        output_format   text | json (default: text)
        additional_args Extra Volatility arguments
    """
    try:
        params = request.json or {}
        image           = params.get("image", "")
        plugin          = params.get("plugin", "")
        pid             = int(params.get("pid", 0))
        output_format   = params.get("output_format", "text")
        additional_args = params.get("additional_args", "")

        if not image:
            return jsonify({"error": "image parameter is required"}), 400
        if not plugin:
            return jsonify({"error": "plugin parameter is required"}), 400
        if not os.path.exists(image):
            return jsonify({"error": f"Memory image not found at path: {image}"}), 400

        valid_formats = ("text", "json")
        if output_format not in valid_formats:
            return jsonify({"error": f"Invalid output_format. Must be one of: {', '.join(valid_formats)}"}), 400

        command = f"vol -f {image}"
        if output_format == "json":
            command += " -r json"
        command += f" {plugin}"
        if pid:
            command += f" --pid {pid}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"volatility {plugin}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/lynis", methods=["POST"])
def lynis():
    """
    Enqueue a Lynis system hardening audit.

    Tier 1 — Low risk. Blue team hardening audit against CIS and security baselines.

    Optional params:
        target          localhost or remote SSH target user@host (default: localhost)
        profile         Lynis profile name (default: default)
        categories      Comma-separated test categories to run
        additional_args Extra Lynis arguments
    """
    try:
        params = request.json or {}
        target          = params.get("target", "localhost")
        profile         = params.get("profile", "default")
        categories      = params.get("categories", "")
        additional_args = params.get("additional_args", "")

        if target == "localhost":
            command = f"lynis audit system --profile {profile} --no-colors"
        else:
            # Remote audit over SSH
            command = f"lynis audit system --profile {profile} --no-colors --remote {target}"

        if categories:
            command += f" --tests-from-group {categories}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"lynis {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/trivy", methods=["POST"])
def trivy():
    """
    Enqueue a Trivy vulnerability and misconfiguration scan.

    Tier 1 — Low risk. Container, filesystem, IaC, and git repo scanning.

    Required params:
        target          Image name, directory path, git URL, or IaC path

    Optional params:
        scan_type       image | fs | repo | config | sbom (default: image)
        severity        Severity filter: LOW,MEDIUM,HIGH,CRITICAL
                        (default: HIGH,CRITICAL)
        ignore_unfixed  Skip unfixed vulnerabilities, bool (default: false)
        output_format   table | json | sarif (default: table)
        additional_args Extra Trivy arguments
    """
    try:
        params = request.json or {}
        target          = params.get("target", "")
        scan_type       = params.get("scan_type", "image")
        severity        = params.get("severity", "HIGH,CRITICAL")
        ignore_unfixed  = params.get("ignore_unfixed", False)
        output_format   = params.get("output_format", "table")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "target parameter is required"}), 400

        valid_scan_types = ("image", "fs", "repo", "config", "sbom")
        if scan_type not in valid_scan_types:
            return jsonify({"error": f"Invalid scan_type. Must be one of: {', '.join(valid_scan_types)}"}), 400

        valid_formats = ("table", "json", "sarif")
        if output_format not in valid_formats:
            return jsonify({"error": f"Invalid output_format. Must be one of: {', '.join(valid_formats)}"}), 400

        command = f"trivy {scan_type} --severity {severity} --format {output_format}"
        if ignore_unfixed:
            command += " --ignore-unfixed"
        if additional_args:
            command += f" {additional_args}"
        command += f" {target}"

        job = enqueue(f"trivy {scan_type} {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/semgrep", methods=["POST"])
def semgrep():
    """
    Enqueue a Semgrep SAST scan.

    Tier 1 — Low risk. Static analysis security testing.
    Finds security issues in source code (C#, TypeScript, Python, etc.).

    Required params:
        path            Path to source directory or file

    Optional params:
        config          Ruleset: auto | p/csharp | p/typescript |
                        p/owasp-top-ten | p/secrets | path to custom rules
                        (default: auto)
        severity        Filter by severity: ERROR | WARNING | INFO
        output_format   text | json | sarif (default: text)
        exclude         Comma-separated glob patterns to exclude
        additional_args Extra Semgrep arguments
    """
    try:
        params = request.json or {}
        path            = params.get("path", "")
        config          = params.get("config", "auto")
        severity        = params.get("severity", "")
        output_format   = params.get("output_format", "text")
        exclude         = params.get("exclude", "")
        additional_args = params.get("additional_args", "")

        if not path:
            return jsonify({"error": "path parameter is required"}), 400
        if not os.path.exists(path):
            return jsonify({"error": f"Path not found: {path}"}), 400

        valid_formats = ("text", "json", "sarif")
        if output_format not in valid_formats:
            return jsonify({"error": f"Invalid output_format. Must be one of: {', '.join(valid_formats)}"}), 400

        command = f"semgrep --config {config}"
        if severity:
            command += f" --severity {severity}"
        if output_format == "json":
            command += " --json"
        elif output_format == "sarif":
            command += " --sarif"
        if exclude:
            for pattern in exclude.split(","):
                command += f" --exclude {pattern.strip()}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {path}"

        job = enqueue(f"semgrep {path}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Health check — also reports queue depth."""
    essential_tools = {
        # Original tools
        "nmap":               "nmap",
        "gobuster":           "gobuster",
        "dirb":               "dirb",
        "nikto":              "nikto",
        "sqlmap":             "sqlmap",
        "wpscan":             "wpscan",
        "enum4linux":         "enum4linux",
        "hydra":              "hydra",
        "john":               "john",
        "msfconsole":         "msfconsole",
        # Red team additions
        "theHarvester":       "theHarvester",
        "amass":              "amass",
        "recon-ng":           "recon-ng",
        "nxc":                "nxc",
        "impacket-secretsdump": "impacket-secretsdump",
        "responder":          "responder",
        "evil-winrm":         "evil-winrm",
        "bloodhound-python":  "bloodhound-python",
        "aircrack-ng":        "aircrack-ng",
        "bettercap":          "bettercap",
        # Blue team additions
        "vol":                "vol",
        "lynis":              "lynis",
        "trivy":              "trivy",
        "semgrep":            "semgrep",
    }

    tools_status = {}
    for label, binary in essential_tools.items():
        try:
            result = run_command(f"which {binary}")
            tools_status[label] = result["success"]
        except Exception:
            tools_status[label] = False

    core_tools   = ["nmap", "gobuster", "dirb", "nikto"]
    all_core_ok  = all(tools_status.get(t, False) for t in core_tools)

    return jsonify({
        "status":                        "healthy",
        "message":                       "Kali Linux Tools API Server is running",
        "tools_status":                  tools_status,
        "all_essential_tools_available": all_core_ok,
        "queue_depth":                   _job_queue.qsize(),
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Kali Linux API Server")
    parser.add_argument("--debug",  action="store_true", help="Enable debug mode")
    parser.add_argument("--port",   type=int, default=API_PORT,    help=f"API port (default: {API_PORT})")
    parser.add_argument("--ip",     type=str, default="127.0.0.1", help="Bind IP (default: 127.0.0.1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        DEBUG_MODE = True
        os.environ["DEBUG_MODE"] = "1"
        logger.setLevel(logging.DEBUG)

    if args.port != API_PORT:
        API_PORT = args.port

    logger.info(f"Starting Kali Linux Tools API Server on {args.ip}:{API_PORT}")
    app.run(host=args.ip, port=API_PORT, debug=DEBUG_MODE)#!/usr/bin/env python3

# This script connect the MCP AI agent to Kali Linux terminal and API Server.

# some of the code here was inspired from https://github.com/whit3rabbit0/project_astro , be sure to check them out

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_PORT = int(os.environ.get("API_PORT", 5000))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0").lower() in ("1", "true", "yes", "y")
COMMAND_TIMEOUT = 600       # 10 minutes — worker-side hard limit
JOB_RETENTION_SECONDS = 3600  # Completed jobs are kept for 1 hour then pruned

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------

class JobStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class JobStore:
    """Thread-safe in-memory store for async job state and results."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, description: str) -> Dict[str, Any]:
        job = {
            "job_id":      job_id,
            "description": description,
            "status":      JobStatus.PENDING,
            "created_at":  time.time(),
            "started_at":  None,
            "finished_at": None,
            "result":      None,
            "error":       None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]     = JobStatus.RUNNING
                self._jobs[job_id]["started_at"] = time.time()

    def set_completed(self, job_id: str, result: Dict[str, Any]):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]      = JobStatus.COMPLETED
                self._jobs[job_id]["finished_at"] = time.time()
                self._jobs[job_id]["result"]      = result

    def set_failed(self, job_id: str, error: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]      = JobStatus.FAILED
                self._jobs[job_id]["finished_at"] = time.time()
                self._jobs[job_id]["error"]       = error

    def list_all(self) -> list:
        with self._lock:
            return list(self._jobs.values())

    def prune_old(self):
        """Remove completed/failed jobs older than JOB_RETENTION_SECONDS."""
        cutoff = time.time() - JOB_RETENTION_SECONDS
        with self._lock:
            stale = [
                jid for jid, job in self._jobs.items()
                if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED)
                and job.get("finished_at", time.time()) < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]
        if stale:
            logger.info(f"Pruned {len(stale)} stale jobs")


job_store = JobStore()
_job_queue: "queue.Queue[tuple]" = queue.Queue()  # (job_id, callable)


def _worker_loop():
    """Background thread: pull jobs from the queue and execute them."""
    logger.info("Job worker thread started")
    while True:
        try:
            job_id, fn = _job_queue.get(timeout=5)
        except queue.Empty:
            job_store.prune_old()
            continue

        job_store.set_running(job_id)
        logger.info(f"[{job_id}] Starting job")
        try:
            result = fn()
            job_store.set_completed(job_id, result)
            logger.info(f"[{job_id}] Job completed (success={result.get('success')})")
        except Exception as exc:
            logger.error(f"[{job_id}] Job raised exception: {exc}")
            logger.error(traceback.format_exc())
            job_store.set_failed(job_id, str(exc))
        finally:
            _job_queue.task_done()


_worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="job-worker")
_worker_thread.start()


def enqueue(description: str, fn: Callable) -> Dict[str, Any]:
    """
    Enqueue a callable for async execution.

    Returns the initial job record (status=pending) immediately so the
    caller can respond with a 202 and a job_id.
    """
    job_id = str(uuid.uuid4())
    job = job_store.create(job_id, description)
    _job_queue.put((job_id, fn))
    logger.info(f"[{job_id}] Enqueued: {description}")
    return job


# ---------------------------------------------------------------------------
# Command Executor (unchanged logic, extracted to reusable helper)
# ---------------------------------------------------------------------------

class CommandExecutor:
    """Handles command execution with graceful timeout management."""

    def __init__(self, command: str, timeout: int = COMMAND_TIMEOUT):
        self.command = command
        self.timeout = timeout
        self.process = None
        self.stdout_data = ""
        self.stderr_data = ""
        self.return_code = None
        self.timed_out = False

    def _read_stdout(self):
        for line in iter(self.process.stdout.readline, ""):
            self.stdout_data += line

    def _read_stderr(self):
        for line in iter(self.process.stderr.readline, ""):
            self.stderr_data += line

    def execute(self) -> Dict[str, Any]:
        logger.info(f"Executing command: {self.command}")
        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            t_out = threading.Thread(target=self._read_stdout, daemon=True)
            t_err = threading.Thread(target=self._read_stderr, daemon=True)
            t_out.start()
            t_err.start()

            try:
                self.return_code = self.process.wait(timeout=self.timeout)
                t_out.join()
                t_err.join()
            except subprocess.TimeoutExpired:
                self.timed_out = True
                logger.warning(f"Command timed out after {self.timeout}s. Terminating.")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Process not responding to SIGTERM — killing.")
                    self.process.kill()
                self.return_code = -1

            success = (
                True if self.timed_out and (self.stdout_data or self.stderr_data)
                else (self.return_code == 0)
            )
            return {
                "stdout":          self.stdout_data,
                "stderr":          self.stderr_data,
                "return_code":     self.return_code,
                "success":         success,
                "timed_out":       self.timed_out,
                "partial_results": self.timed_out and bool(self.stdout_data or self.stderr_data),
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            logger.error(traceback.format_exc())
            return {
                "stdout":          self.stdout_data,
                "stderr":          f"Error: {e}\n{self.stderr_data}",
                "return_code":     -1,
                "success":         False,
                "timed_out":       False,
                "partial_results": bool(self.stdout_data or self.stderr_data),
            }


def run_command(command: str) -> Dict[str, Any]:
    return CommandExecutor(command).execute()


# ---------------------------------------------------------------------------
# Helper: build a quick 202 response
# ---------------------------------------------------------------------------

def accepted(job: Dict[str, Any]):
    """Return HTTP 202 Accepted with job metadata."""
    return jsonify({
        "job_id":      job["job_id"],
        "status":      job["status"],
        "description": job["description"],
    }), 202


# ---------------------------------------------------------------------------
# Job status / management endpoints
# ---------------------------------------------------------------------------

@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    """Poll the status (and result) of an async job."""
    job = job_store.get(job_id)
    if job is None:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404
    return jsonify(job)


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """List all tracked jobs (useful for debugging)."""
    return jsonify(job_store.list_all())


# ---------------------------------------------------------------------------
# Tool endpoints — all return 202 + job_id immediately
# ---------------------------------------------------------------------------

@app.route("/api/command", methods=["POST"])
def generic_command():
    """Enqueue an arbitrary shell command."""
    try:
        params = request.json or {}
        command = params.get("command", "")
        if not command:
            return jsonify({"error": "Command parameter is required"}), 400

        job = enqueue(f"command: {command[:80]}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/nmap", methods=["POST"])
def nmap():
    """Enqueue an nmap scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        scan_type       = params.get("scan_type", "-sCV")
        ports           = params.get("ports", "")
        additional_args = params.get("additional_args", "-T4 -Pn")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"nmap {scan_type}"
        if ports:
            command += f" -p {ports}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {target}"

        job = enqueue(f"nmap {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/gobuster", methods=["POST"])
def gobuster():
    """Enqueue a gobuster scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        mode            = params.get("mode", "dir")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400
        if mode not in ("dir", "dns", "fuzz", "vhost"):
            return jsonify({"error": f"Invalid mode: {mode}"}), 400

        command = f"gobuster {mode} -u {url} -w {wordlist}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"gobuster {mode} {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/dirb", methods=["POST"])
def dirb():
    """Enqueue a dirb scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"dirb {url} {wordlist}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"dirb {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/nikto", methods=["POST"])
def nikto():
    """Enqueue a nikto scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"nikto -h {target}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"nikto {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/sqlmap", methods=["POST"])
def sqlmap():
    """Enqueue a sqlmap scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        post_data       = params.get("data", "")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"sqlmap -u {url} --batch"
        if post_data:
            command += f" --data={post_data}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"sqlmap {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/metasploit", methods=["POST"])
def metasploit():
    """Enqueue a Metasploit module run."""
    try:
        params = request.json or {}
        module  = params.get("module", "")
        options = params.get("options", {})

        if not module:
            return jsonify({"error": "Module parameter is required"}), 400

        resource_file = f"/tmp/msf_{uuid.uuid4().hex}.rc"
        resource_lines = [f"use {module}"]
        for key, value in options.items():
            resource_lines.append(f"set {key} {value}")
        resource_lines.append("run")
        resource_content = "\n".join(resource_lines) + "\n"

        with open(resource_file, "w") as f:
            f.write(resource_content)

        def _run(cmd=f"msfconsole -q -r {resource_file}", rc=resource_file):
            result = run_command(cmd)
            try:
                os.remove(rc)
            except Exception:
                pass
            return result

        job = enqueue(f"metasploit {module}", _run)
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/hydra", methods=["POST"])
def hydra():
    """Enqueue a Hydra password attack."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        service         = params.get("service", "")
        username        = params.get("username", "")
        username_file   = params.get("username_file", "")
        password        = params.get("password", "")
        password_file   = params.get("password_file", "")
        additional_args = params.get("additional_args", "")

        if not target or not service:
            return jsonify({"error": "Target and service parameters are required"}), 400
        if not (username or username_file) or not (password or password_file):
            return jsonify({"error": "Username/username_file and password/password_file are required"}), 400

        command = "hydra -t 4"
        command += f" -l {username}" if username else f" -L {username_file}"
        command += f" -p {password}" if password else f" -P {password_file}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {target} {service}"

        job = enqueue(f"hydra {service}@{target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/john", methods=["POST"])
def john():
    """Enqueue a John the Ripper cracking job."""
    try:
        params = request.json or {}
        hash_file       = params.get("hash_file", "")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/rockyou.txt")
        format_type     = params.get("format", "")
        additional_args = params.get("additional_args", "")

        if not hash_file:
            return jsonify({"error": "Hash file parameter is required"}), 400

        command = "john"
        if format_type:
            command += f" --format={format_type}"
        if wordlist:
            command += f" --wordlist={wordlist}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {hash_file}"

        job = enqueue(f"john {hash_file}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/wpscan", methods=["POST"])
def wpscan():
    """Enqueue a WPScan scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"wpscan --url {url}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"wpscan {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/enum4linux", methods=["POST"])
def enum4linux():
    """Enqueue an enum4linux scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        additional_args = params.get("additional_args", "-a")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"enum4linux {additional_args} {target}"

        job = enqueue(f"enum4linux {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Red Team Tool Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/tools/theharvester", methods=["POST"])
def theharvester():
    """
    Enqueue a theHarvester OSINT scan.

    Tier 1 — Low risk. Passive OSINT only.
    Harvests emails, subdomains, hosts, and employee names from public sources.

    Required params:
        domain          Target domain to harvest

    Optional params:
        sources         Comma-separated data sources (default: broad passive set)
        limit           Max results to retrieve (default: 500)
        dns_resolve     Resolve discovered hosts, bool (default: false)
        additional_args Extra theHarvester arguments
    """
    try:
        params = request.json or {}
        domain          = params.get("domain", "")
        sources         = params.get("sources", "baidu,bing,crtsh,dnsdumpster,duckduckgo,hackertarget,rapiddns,urlscan")
        limit           = int(params.get("limit", 500))
        dns_resolve     = params.get("dns_resolve", False)
        additional_args = params.get("additional_args", "")

        if not domain:
            return jsonify({"error": "domain parameter is required"}), 400

        command = f"theHarvester -d {domain} -b {sources} -l {limit}"
        if dns_resolve:
            command += " -r"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"theharvester {domain}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/amass", methods=["POST"])
def amass():
    """
    Enqueue an Amass subdomain enumeration.

    Tier 1 (passive) / Tier 2 (active/brute).
    Deep subdomain enumeration using passive and active techniques.

    Required params:
        domain          Target domain

    Optional params:
        mode            passive | active | brute (default: passive)
        wordlist        Path to wordlist — used when mode is 'brute'
        output_dir      Directory for Amass output files (default: /tmp/amass_out)
        additional_args Extra Amass arguments
    """
    try:
        params = request.json or {}
        domain          = params.get("domain", "")
        mode            = params.get("mode", "passive")
        wordlist        = params.get("wordlist", "")
        output_dir      = params.get("output_dir", "/tmp/amass_out")
        additional_args = params.get("additional_args", "")

        if not domain:
            return jsonify({"error": "domain parameter is required"}), 400
        if mode not in ("passive", "active", "brute"):
            return jsonify({"error": f"Invalid mode '{mode}'. Must be: passive, active, brute"}), 400
        if mode == "brute" and not wordlist:
            return jsonify({"error": "wordlist is required when mode is 'brute'"}), 400

        command = f"amass enum -{mode} -d {domain} -dir {output_dir}"
        if mode == "brute" and wordlist:
            command += f" -brute -w {wordlist}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"amass {mode} {domain}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/recon-ng", methods=["POST"])
def recon_ng():
    """
    Enqueue a Recon-ng module run.

    Tier 1 — Low risk. Modular OSINT framework.

    Required params:
        module          Full module path e.g. recon/domains-hosts/hackertarget
        source          Source value (domain, IP, company name, etc.)

    Optional params:
        workspace       Recon-ng workspace name (default: default)
        additional_args Extra key=value options passed to the module
    """
    try:
        params = request.json or {}
        module          = params.get("module", "")
        source          = params.get("source", "")
        workspace       = params.get("workspace", "default")
        additional_args = params.get("additional_args", "")

        if not module:
            return jsonify({"error": "module parameter is required"}), 400
        if not source:
            return jsonify({"error": "source parameter is required"}), 400

        # Build a recon-ng script file for non-interactive execution
        script_file = f"/tmp/recon_ng_{uuid.uuid4().hex}.rc"
        script_lines = [
            f"workspaces select {workspace}",
            f"modules load {module}",
            f"options set SOURCE {source}",
        ]
        if additional_args:
            for opt in additional_args.split(","):
                opt = opt.strip()
                if "=" in opt:
                    k, v = opt.split("=", 1)
                    script_lines.append(f"options set {k.strip()} {v.strip()}")
        script_lines.append("run")
        script_lines.append("exit")

        with open(script_file, "w") as f:
            f.write("\n".join(script_lines) + "\n")

        def _run(cmd=f"recon-ng -r {script_file}", rc=script_file):
            result = run_command(cmd)
            try:
                os.remove(rc)
            except Exception:
                pass
            return result

        job = enqueue(f"recon-ng {module} ({source})", _run)
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/netexec", methods=["POST"])
def netexec():
    """
    Enqueue a NetExec (nxc) lateral movement/enumeration run.

    Tier 2–3. SMB, WinRM, LDAP, RDP, SSH testing across Windows environments.

    Required params:
        target          IP, CIDR, or hostname
        protocol        smb | winrm | ldap | rdp | ssh | mssql

    Optional params:
        username        Single username
        username_file   Path to username list
        password        Single password
        password_file   Path to password list
        hash            NTLM hash for Pass-the-Hash (format: LMHASH:NTHASH or just NTHASH)
        action          Module/action flag e.g. --shares, --users, --sam, --lsa
        domain          Windows domain name
        additional_args Extra netexec arguments
    """
    try:
        params = request.json or {}
        target          = params.get("target", "")
        protocol        = params.get("protocol", "")
        username        = params.get("username", "")
        username_file   = params.get("username_file", "")
        password        = params.get("password", "")
        password_file   = params.get("password_file", "")
        ntlm_hash       = params.get("hash", "")
        action          = params.get("action", "")
        domain          = params.get("domain", "")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "target parameter is required"}), 400
        if not protocol:
            return jsonify({"error": "protocol parameter is required"}), 400

        valid_protocols = ("smb", "winrm", "ldap", "rdp", "ssh", "mssql")
        if protocol not in valid_protocols:
            return jsonify({"error": f"Invalid protocol '{protocol}'. Must be one of: {', '.join(valid_protocols)}"}), 400

        command = f"nxc {protocol} {target}"
        if domain:
            command += f" -d {domain}"
        if username:
            command += f" -u {username}"
        elif username_file:
            command += f" -U {username_file}"
        if ntlm_hash:
            command += f" -H {ntlm_hash}"
        elif password:
            command += f" -p {password}"
        elif password_file:
            command += f" -P {password_file}"
        if action:
            command += f" {action}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"netexec {protocol} {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/impacket", methods=["POST"])
def impacket():
    """
    Enqueue an Impacket script execution.

    Tier 3 — High risk. Windows/AD protocol abuse.
    Use only on explicitly authorized targets.

    Required params:
        script          Script name: secretsdump | GetNPUsers | GetUserSPNs |
                        psexec | wmiexec | smbclient | ticketer | getPac
        target          Target IP or hostname

    Optional params:
        username        Username
        password        Password
        hash            NTLM hash (format: LMHASH:NTHASH or just NTHASH)
        domain          Domain name
        dc_ip           Domain controller IP
        additional_args Extra arguments for the specific script
    """
    try:
        params = request.json or {}
        script          = params.get("script", "")
        target          = params.get("target", "")
        username        = params.get("username", "")
        password        = params.get("password", "")
        ntlm_hash       = params.get("hash", "")
        domain          = params.get("domain", "")
        dc_ip           = params.get("dc_ip", "")
        additional_args = params.get("additional_args", "")

        valid_scripts = (
            "secretsdump", "GetNPUsers", "GetUserSPNs",
            "psexec", "wmiexec", "smbclient", "ticketer", "getPac",
        )
        if not script:
            return jsonify({"error": "script parameter is required"}), 400
        if script not in valid_scripts:
            return jsonify({"error": f"Invalid script '{script}'. Must be one of: {', '.join(valid_scripts)}"}), 400
        if not target:
            return jsonify({"error": "target parameter is required"}), 400

        # Build auth string: DOMAIN/user:pass@target  or  DOMAIN/user@target (hash separately)
        auth_user = f"{domain}\\{username}" if domain and username else username
        if password and not ntlm_hash:
            auth_string = f"{auth_user}:{password}@{target}" if auth_user else f"@{target}"
        else:
            auth_string = f"{auth_user}@{target}" if auth_user else f"@{target}"

        command = f"impacket-{script} {auth_string}"
        if ntlm_hash:
            # Impacket expects LMHASH:NTHASH — prepend empty LM if only NT provided
            hash_val = ntlm_hash if ":" in ntlm_hash else f":{ntlm_hash}"
            command += f" -hashes {hash_val}"
        if dc_ip:
            command += f" -dc-ip {dc_ip}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"impacket-{script} {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/responder", methods=["POST"])
def responder():
    """
    Enqueue a Responder LLMNR/NBT-NS/mDNS poisoning run.

    Tier 3 — High risk. Captures NTLMv2 hashes on internal segments.
    Use only on explicitly authorized internal network segments.

    Required params:
        interface       Network interface to listen on (e.g. eth0)

    Optional params:
        duration_seconds  How long to run before auto-stopping (default: 60)
        modes           Extra mode flags e.g. --wpad --lm (default: --wpad)
        additional_args Extra Responder arguments
    """
    try:
        params = request.json or {}
        interface        = params.get("interface", "")
        duration_seconds = int(params.get("duration_seconds", 60))
        modes            = params.get("modes", "--wpad")
        additional_args  = params.get("additional_args", "")

        if not interface:
            return jsonify({"error": "interface parameter is required"}), 400
        if duration_seconds < 1 or duration_seconds > 3600:
            return jsonify({"error": "duration_seconds must be between 1 and 3600"}), 400

        command = f"timeout {duration_seconds} responder -I {interface} {modes}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"responder {interface}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/evil-winrm", methods=["POST"])
def evil_winrm():
    """
    Enqueue an Evil-WinRM command execution.

    Tier 3 — High risk. WinRM post-exploitation shell access.
    Runs a single non-interactive command rather than an interactive shell.

    Required params:
        target          Target IP or hostname
        username        Username
        command         Command to execute on the remote host

    Optional params:
        password        Password (use either password or hash, not both)
        hash            NTLM hash for Pass-the-Hash
        additional_args Extra Evil-WinRM arguments
    """
    try:
        params = request.json or {}
        target          = params.get("target", "")
        username        = params.get("username", "")
        password        = params.get("password", "")
        ntlm_hash       = params.get("hash", "")
        command_to_run  = params.get("command", "")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "target parameter is required"}), 400
        if not username:
            return jsonify({"error": "username parameter is required"}), 400
        if not command_to_run:
            return jsonify({"error": "command parameter is required"}), 400
        if not password and not ntlm_hash:
            return jsonify({"error": "Either password or hash is required"}), 400

        command = f"evil-winrm -i {target} -u {username}"
        if ntlm_hash:
            command += f" -H {ntlm_hash}"
        elif password:
            command += f" -p {password}"
        # Execute a single command via -e flag (non-interactive)
        command += f" -e \"{command_to_run}\""
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"evil-winrm {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/bloodhound", methods=["POST"])
def bloodhound():
    """
    Enqueue a BloodHound data collection run (bloodhound-python).

    Tier 2 — Medium risk. Collects AD data for BloodHound attack path analysis.
    Collection only — visualization requires the BloodHound UI separately.

    Required params:
        domain          Target domain
        dc_ip           Domain controller IP
        username        Domain username

    Optional params:
        password            Password
        hash                NTLM hash (format: LMHASH:NTHASH or just NTHASH)
        collection_methods  Default | All | DCOnly | Session | Trusts | ACL | Group
                            (default: Default)
        output_dir          Directory for JSON output files (default: /tmp/bh_out)
        additional_args     Extra bloodhound-python arguments
    """
    try:
        params = request.json or {}
        domain              = params.get("domain", "")
        dc_ip               = params.get("dc_ip", "")
        username            = params.get("username", "")
        password            = params.get("password", "")
        ntlm_hash           = params.get("hash", "")
        collection_methods  = params.get("collection_methods", "Default")
        output_dir          = params.get("output_dir", "/tmp/bh_out")
        additional_args     = params.get("additional_args", "")

        if not domain:
            return jsonify({"error": "domain parameter is required"}), 400
        if not dc_ip:
            return jsonify({"error": "dc_ip parameter is required"}), 400
        if not username:
            return jsonify({"error": "username parameter is required"}), 400
        if not password and not ntlm_hash:
            return jsonify({"error": "Either password or hash is required"}), 400

        valid_methods = ("Default", "All", "DCOnly", "Session", "Trusts", "ACL", "Group")
        if collection_methods not in valid_methods:
            return jsonify({"error": f"Invalid collection_methods. Must be one of: {', '.join(valid_methods)}"}), 400

        os.makedirs(output_dir, exist_ok=True)

        command = (
            f"bloodhound-python -d {domain} -u {username} -ns {dc_ip}"
            f" -c {collection_methods} -o {output_dir} --zip"
        )
        if ntlm_hash:
            hash_val = ntlm_hash if ":" in ntlm_hash else f":{ntlm_hash}"
            command += f" --hashes {hash_val}"
        elif password:
            command += f" -p {password}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"bloodhound-python {domain}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/aircrack", methods=["POST"])
def aircrack():
    """
    Enqueue an aircrack-ng Wi-Fi action.

    Tier 3 — High risk. Wi-Fi assessment (monitor mode, capture, deauth, crack).
    Use only on networks you are explicitly authorized to test.

    Required params:
        action          monitor_start | capture | deauth | crack
        interface       Wireless interface (e.g. wlan0)

    Optional params (by action):
        bssid           Target AP BSSID — required for capture/deauth
        channel         Channel to lock on — used for capture (default: 0 = auto)
        capture_file    Output/input .cap file path (default: /tmp/capture)
        wordlist        Wordlist path — used for crack (default: rockyou.txt)
        duration_seconds  Capture run duration in seconds (default: 60)
        additional_args Extra arguments
    """
    try:
        params = request.json or {}
        action           = params.get("action", "")
        interface        = params.get("interface", "")
        bssid            = params.get("bssid", "")
        channel          = int(params.get("channel", 0))
        capture_file     = params.get("capture_file", "/tmp/capture")
        wordlist         = params.get("wordlist", "/usr/share/wordlists/rockyou.txt")
        duration_seconds = int(params.get("duration_seconds", 60))
        additional_args  = params.get("additional_args", "")

        valid_actions = ("monitor_start", "capture", "deauth", "crack")
        if not action:
            return jsonify({"error": "action parameter is required"}), 400
        if action not in valid_actions:
            return jsonify({"error": f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"}), 400
        if not interface:
            return jsonify({"error": "interface parameter is required"}), 400

        if action == "monitor_start":
            command = f"airmon-ng start {interface}"
        elif action == "capture":
            if not bssid:
                return jsonify({"error": "bssid is required for capture action"}), 400
            command = f"timeout {duration_seconds} airodump-ng"
            command += f" --bssid {bssid}"
            if channel:
                command += f" --channel {channel}"
            command += f" -w {capture_file} {interface}mon"
        elif action == "deauth":
            if not bssid:
                return jsonify({"error": "bssid is required for deauth action"}), 400
            command = f"aireplay-ng --deauth 10 -a {bssid} {interface}mon"
        elif action == "crack":
            command = f"aircrack-ng {capture_file}*.cap -w {wordlist}"

        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"aircrack {action} {interface}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/bettercap", methods=["POST"])
def bettercap():
    """
    Enqueue a Bettercap network attack/recon run.

    Tier 3 — High risk. Network recon, MITM, ARP spoofing, Wi-Fi, and BLE.
    Use only on explicitly authorized network segments.

    Required params:
        interface       Network interface (e.g. eth0)

    Optional params:
        caplet          Path to a .cap caplet script file
        commands        Semicolon-separated bettercap commands to evaluate
        duration_seconds  Run duration before auto-stop (default: 60)
        additional_args Extra bettercap arguments
    """
    try:
        params = request.json or {}
        interface        = params.get("interface", "")
        caplet           = params.get("caplet", "")
        commands         = params.get("commands", "")
        duration_seconds = int(params.get("duration_seconds", 60))
        additional_args  = params.get("additional_args", "")

        if not interface:
            return jsonify({"error": "interface parameter is required"}), 400
        if not caplet and not commands:
            return jsonify({"error": "Either caplet or commands is required"}), 400
        if duration_seconds < 1 or duration_seconds > 3600:
            return jsonify({"error": "duration_seconds must be between 1 and 3600"}), 400

        command = f"timeout {duration_seconds} bettercap -iface {interface}"
        if caplet:
            command += f" -caplet {caplet}"
        if commands:
            command += f" -eval \"{commands}\""
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"bettercap {interface}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Blue Team Tool Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/tools/volatility", methods=["POST"])
def volatility():
    """
    Enqueue a Volatility 3 memory forensics run.

    Tier 1 — Low risk. Forensic analysis of memory dumps.
    No active network testing — safe for incident response workflows.

    Required params:
        image           Path to memory dump file
        plugin          Plugin to run e.g. windows.pslist, windows.netscan,
                        windows.malfind, windows.hashdump, windows.cmdline,
                        windows.dlllist, windows.dumpfiles, linux.pslist,
                        linux.bash

    Optional params:
        pid             Filter by PID (0 = all, default: 0)
        output_format   text | json (default: text)
        additional_args Extra Volatility arguments
    """
    try:
        params = request.json or {}
        image           = params.get("image", "")
        plugin          = params.get("plugin", "")
        pid             = int(params.get("pid", 0))
        output_format   = params.get("output_format", "text")
        additional_args = params.get("additional_args", "")

        if not image:
            return jsonify({"error": "image parameter is required"}), 400
        if not plugin:
            return jsonify({"error": "plugin parameter is required"}), 400
        if not os.path.exists(image):
            return jsonify({"error": f"Memory image not found at path: {image}"}), 400

        valid_formats = ("text", "json")
        if output_format not in valid_formats:
            return jsonify({"error": f"Invalid output_format. Must be one of: {', '.join(valid_formats)}"}), 400

        command = f"vol -f {image}"
        if output_format == "json":
            command += " -r json"
        command += f" {plugin}"
        if pid:
            command += f" --pid {pid}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"volatility {plugin}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/lynis", methods=["POST"])
def lynis():
    """
    Enqueue a Lynis system hardening audit.

    Tier 1 — Low risk. Blue team hardening audit against CIS and security baselines.

    Optional params:
        target          localhost or remote SSH target user@host (default: localhost)
        profile         Lynis profile name (default: default)
        categories      Comma-separated test categories to run
        additional_args Extra Lynis arguments
    """
    try:
        params = request.json or {}
        target          = params.get("target", "localhost")
        profile         = params.get("profile", "default")
        categories      = params.get("categories", "")
        additional_args = params.get("additional_args", "")

        if target == "localhost":
            command = f"lynis audit system --profile {profile} --no-colors"
        else:
            # Remote audit over SSH
            command = f"lynis audit system --profile {profile} --no-colors --remote {target}"

        if categories:
            command += f" --tests-from-group {categories}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"lynis {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/trivy", methods=["POST"])
def trivy():
    """
    Enqueue a Trivy vulnerability and misconfiguration scan.

    Tier 1 — Low risk. Container, filesystem, IaC, and git repo scanning.

    Required params:
        target          Image name, directory path, git URL, or IaC path

    Optional params:
        scan_type       image | fs | repo | config | sbom (default: image)
        severity        Severity filter: LOW,MEDIUM,HIGH,CRITICAL
                        (default: HIGH,CRITICAL)
        ignore_unfixed  Skip unfixed vulnerabilities, bool (default: false)
        output_format   table | json | sarif (default: table)
        additional_args Extra Trivy arguments
    """
    try:
        params = request.json or {}
        target          = params.get("target", "")
        scan_type       = params.get("scan_type", "image")
        severity        = params.get("severity", "HIGH,CRITICAL")
        ignore_unfixed  = params.get("ignore_unfixed", False)
        output_format   = params.get("output_format", "table")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "target parameter is required"}), 400

        valid_scan_types = ("image", "fs", "repo", "config", "sbom")
        if scan_type not in valid_scan_types:
            return jsonify({"error": f"Invalid scan_type. Must be one of: {', '.join(valid_scan_types)}"}), 400

        valid_formats = ("table", "json", "sarif")
        if output_format not in valid_formats:
            return jsonify({"error": f"Invalid output_format. Must be one of: {', '.join(valid_formats)}"}), 400

        command = f"trivy {scan_type} --severity {severity} --format {output_format}"
        if ignore_unfixed:
            command += " --ignore-unfixed"
        if additional_args:
            command += f" {additional_args}"
        command += f" {target}"

        job = enqueue(f"trivy {scan_type} {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/semgrep", methods=["POST"])
def semgrep():
    """
    Enqueue a Semgrep SAST scan.

    Tier 1 — Low risk. Static analysis security testing.
    Finds security issues in source code (C#, TypeScript, Python, etc.).

    Required params:
        path            Path to source directory or file

    Optional params:
        config          Ruleset: auto | p/csharp | p/typescript |
                        p/owasp-top-ten | p/secrets | path to custom rules
                        (default: auto)
        severity        Filter by severity: ERROR | WARNING | INFO
        output_format   text | json | sarif (default: text)
        exclude         Comma-separated glob patterns to exclude
        additional_args Extra Semgrep arguments
    """
    try:
        params = request.json or {}
        path            = params.get("path", "")
        config          = params.get("config", "auto")
        severity        = params.get("severity", "")
        output_format   = params.get("output_format", "text")
        exclude         = params.get("exclude", "")
        additional_args = params.get("additional_args", "")

        if not path:
            return jsonify({"error": "path parameter is required"}), 400
        if not os.path.exists(path):
            return jsonify({"error": f"Path not found: {path}"}), 400

        valid_formats = ("text", "json", "sarif")
        if output_format not in valid_formats:
            return jsonify({"error": f"Invalid output_format. Must be one of: {', '.join(valid_formats)}"}), 400

        command = f"semgrep --config {config}"
        if severity:
            command += f" --severity {severity}"
        if output_format == "json":
            command += " --json"
        elif output_format == "sarif":
            command += " --sarif"
        if exclude:
            for pattern in exclude.split(","):
                command += f" --exclude {pattern.strip()}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {path}"

        job = enqueue(f"semgrep {path}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Health check — also reports queue depth."""
    essential_tools = {
        # Original tools
        "nmap":               "nmap",
        "gobuster":           "gobuster",
        "dirb":               "dirb",
        "nikto":              "nikto",
        "sqlmap":             "sqlmap",
        "wpscan":             "wpscan",
        "enum4linux":         "enum4linux",
        "hydra":              "hydra",
        "john":               "john",
        "msfconsole":         "msfconsole",
        # Red team additions
        "theHarvester":       "theHarvester",
        "amass":              "amass",
        "recon-ng":           "recon-ng",
        "nxc":                "nxc",
        "impacket-secretsdump": "impacket-secretsdump",
        "responder":          "responder",
        "evil-winrm":         "evil-winrm",
        "bloodhound-python":  "bloodhound-python",
        "aircrack-ng":        "aircrack-ng",
        "bettercap":          "bettercap",
        # Blue team additions
        "vol":                "vol",
        "lynis":              "lynis",
        "trivy":              "trivy",
        "semgrep":            "semgrep",
    }

    tools_status = {}
    for label, binary in essential_tools.items():
        try:
            result = run_command(f"which {binary}")
            tools_status[label] = result["success"]
        except Exception:
            tools_status[label] = False

    core_tools   = ["nmap", "gobuster", "dirb", "nikto"]
    all_core_ok  = all(tools_status.get(t, False) for t in core_tools)

    return jsonify({
        "status":                        "healthy",
        "message":                       "Kali Linux Tools API Server is running",
        "tools_status":                  tools_status,
        "all_essential_tools_available": all_core_ok,
        "queue_depth":                   _job_queue.qsize(),
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Kali Linux API Server")
    parser.add_argument("--debug",  action="store_true", help="Enable debug mode")
    parser.add_argument("--port",   type=int, default=API_PORT,    help=f"API port (default: {API_PORT})")
    parser.add_argument("--ip",     type=str, default="127.0.0.1", help="Bind IP (default: 127.0.0.1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        DEBUG_MODE = True
        os.environ["DEBUG_MODE"] = "1"
        logger.setLevel(logging.DEBUG)

    if args.port != API_PORT:
        API_PORT = args.port

    logger.info(f"Starting Kali Linux Tools API Server on {args.ip}:{API_PORT}")
    app.run(host=args.ip, port=API_PORT, debug=DEBUG_MODE)#!/usr/bin/env python3

# This script connect the MCP AI agent to Kali Linux terminal and API Server.

# some of the code here was inspired from https://github.com/whit3rabbit0/project_astro , be sure to check them out

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_PORT = int(os.environ.get("API_PORT", 5000))
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0").lower() in ("1", "true", "yes", "y")
COMMAND_TIMEOUT = 600       # 10 minutes — worker-side hard limit
JOB_RETENTION_SECONDS = 3600  # Completed jobs are kept for 1 hour then pruned

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Job Queue
# ---------------------------------------------------------------------------

class JobStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class JobStore:
    """Thread-safe in-memory store for async job state and results."""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, description: str) -> Dict[str, Any]:
        job = {
            "job_id":      job_id,
            "description": description,
            "status":      JobStatus.PENDING,
            "created_at":  time.time(),
            "started_at":  None,
            "finished_at": None,
            "result":      None,
            "error":       None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_running(self, job_id: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]     = JobStatus.RUNNING
                self._jobs[job_id]["started_at"] = time.time()

    def set_completed(self, job_id: str, result: Dict[str, Any]):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]      = JobStatus.COMPLETED
                self._jobs[job_id]["finished_at"] = time.time()
                self._jobs[job_id]["result"]      = result

    def set_failed(self, job_id: str, error: str):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"]      = JobStatus.FAILED
                self._jobs[job_id]["finished_at"] = time.time()
                self._jobs[job_id]["error"]       = error

    def list_all(self) -> list:
        with self._lock:
            return list(self._jobs.values())

    def prune_old(self):
        """Remove completed/failed jobs older than JOB_RETENTION_SECONDS."""
        cutoff = time.time() - JOB_RETENTION_SECONDS
        with self._lock:
            stale = [
                jid for jid, job in self._jobs.items()
                if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED)
                and job.get("finished_at", time.time()) < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]
        if stale:
            logger.info(f"Pruned {len(stale)} stale jobs")


job_store = JobStore()
_job_queue: "queue.Queue[tuple]" = queue.Queue()  # (job_id, callable)


def _worker_loop():
    """Background thread: pull jobs from the queue and execute them."""
    logger.info("Job worker thread started")
    while True:
        try:
            job_id, fn = _job_queue.get(timeout=5)
        except queue.Empty:
            job_store.prune_old()
            continue

        job_store.set_running(job_id)
        logger.info(f"[{job_id}] Starting job")
        try:
            result = fn()
            job_store.set_completed(job_id, result)
            logger.info(f"[{job_id}] Job completed (success={result.get('success')})")
        except Exception as exc:
            logger.error(f"[{job_id}] Job raised exception: {exc}")
            logger.error(traceback.format_exc())
            job_store.set_failed(job_id, str(exc))
        finally:
            _job_queue.task_done()


_worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="job-worker")
_worker_thread.start()


def enqueue(description: str, fn: Callable) -> Dict[str, Any]:
    """
    Enqueue a callable for async execution.

    Returns the initial job record (status=pending) immediately so the
    caller can respond with a 202 and a job_id.
    """
    job_id = str(uuid.uuid4())
    job = job_store.create(job_id, description)
    _job_queue.put((job_id, fn))
    logger.info(f"[{job_id}] Enqueued: {description}")
    return job


# ---------------------------------------------------------------------------
# Command Executor (unchanged logic, extracted to reusable helper)
# ---------------------------------------------------------------------------

class CommandExecutor:
    """Handles command execution with graceful timeout management."""

    def __init__(self, command: str, timeout: int = COMMAND_TIMEOUT):
        self.command = command
        self.timeout = timeout
        self.process = None
        self.stdout_data = ""
        self.stderr_data = ""
        self.return_code = None
        self.timed_out = False

    def _read_stdout(self):
        for line in iter(self.process.stdout.readline, ""):
            self.stdout_data += line

    def _read_stderr(self):
        for line in iter(self.process.stderr.readline, ""):
            self.stderr_data += line

    def execute(self) -> Dict[str, Any]:
        logger.info(f"Executing command: {self.command}")
        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            t_out = threading.Thread(target=self._read_stdout, daemon=True)
            t_err = threading.Thread(target=self._read_stderr, daemon=True)
            t_out.start()
            t_err.start()

            try:
                self.return_code = self.process.wait(timeout=self.timeout)
                t_out.join()
                t_err.join()
            except subprocess.TimeoutExpired:
                self.timed_out = True
                logger.warning(f"Command timed out after {self.timeout}s. Terminating.")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Process not responding to SIGTERM — killing.")
                    self.process.kill()
                self.return_code = -1

            success = (
                True if self.timed_out and (self.stdout_data or self.stderr_data)
                else (self.return_code == 0)
            )
            return {
                "stdout":          self.stdout_data,
                "stderr":          self.stderr_data,
                "return_code":     self.return_code,
                "success":         success,
                "timed_out":       self.timed_out,
                "partial_results": self.timed_out and bool(self.stdout_data or self.stderr_data),
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            logger.error(traceback.format_exc())
            return {
                "stdout":          self.stdout_data,
                "stderr":          f"Error: {e}\n{self.stderr_data}",
                "return_code":     -1,
                "success":         False,
                "timed_out":       False,
                "partial_results": bool(self.stdout_data or self.stderr_data),
            }


def run_command(command: str) -> Dict[str, Any]:
    return CommandExecutor(command).execute()


# ---------------------------------------------------------------------------
# Helper: build a quick 202 response
# ---------------------------------------------------------------------------

def accepted(job: Dict[str, Any]):
    """Return HTTP 202 Accepted with job metadata."""
    return jsonify({
        "job_id":      job["job_id"],
        "status":      job["status"],
        "description": job["description"],
    }), 202


# ---------------------------------------------------------------------------
# Job status / management endpoints
# ---------------------------------------------------------------------------

@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    """Poll the status (and result) of an async job."""
    job = job_store.get(job_id)
    if job is None:
        return jsonify({"error": f"Job '{job_id}' not found"}), 404
    return jsonify(job)


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    """List all tracked jobs (useful for debugging)."""
    return jsonify(job_store.list_all())


# ---------------------------------------------------------------------------
# Tool endpoints — all return 202 + job_id immediately
# ---------------------------------------------------------------------------

@app.route("/api/command", methods=["POST"])
def generic_command():
    """Enqueue an arbitrary shell command."""
    try:
        params = request.json or {}
        command = params.get("command", "")
        if not command:
            return jsonify({"error": "Command parameter is required"}), 400

        job = enqueue(f"command: {command[:80]}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/nmap", methods=["POST"])
def nmap():
    """Enqueue an nmap scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        scan_type       = params.get("scan_type", "-sCV")
        ports           = params.get("ports", "")
        additional_args = params.get("additional_args", "-T4 -Pn")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"nmap {scan_type}"
        if ports:
            command += f" -p {ports}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {target}"

        job = enqueue(f"nmap {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/gobuster", methods=["POST"])
def gobuster():
    """Enqueue a gobuster scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        mode            = params.get("mode", "dir")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400
        if mode not in ("dir", "dns", "fuzz", "vhost"):
            return jsonify({"error": f"Invalid mode: {mode}"}), 400

        command = f"gobuster {mode} -u {url} -w {wordlist}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"gobuster {mode} {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/dirb", methods=["POST"])
def dirb():
    """Enqueue a dirb scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/dirb/common.txt")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"dirb {url} {wordlist}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"dirb {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/nikto", methods=["POST"])
def nikto():
    """Enqueue a nikto scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        additional_args = params.get("additional_args", "")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"nikto -h {target}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"nikto {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/sqlmap", methods=["POST"])
def sqlmap():
    """Enqueue a sqlmap scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        post_data       = params.get("data", "")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"sqlmap -u {url} --batch"
        if post_data:
            command += f" --data={post_data}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"sqlmap {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/metasploit", methods=["POST"])
def metasploit():
    """Enqueue a Metasploit module run."""
    try:
        params = request.json or {}
        module  = params.get("module", "")
        options = params.get("options", {})

        if not module:
            return jsonify({"error": "Module parameter is required"}), 400

        resource_file = f"/tmp/msf_{uuid.uuid4().hex}.rc"
        resource_lines = [f"use {module}"]
        for key, value in options.items():
            resource_lines.append(f"set {key} {value}")
        resource_lines.append("run")
        resource_content = "\n".join(resource_lines) + "\n"

        with open(resource_file, "w") as f:
            f.write(resource_content)

        def _run(cmd=f"msfconsole -q -r {resource_file}", rc=resource_file):
            result = run_command(cmd)
            try:
                os.remove(rc)
            except Exception:
                pass
            return result

        job = enqueue(f"metasploit {module}", _run)
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/hydra", methods=["POST"])
def hydra():
    """Enqueue a Hydra password attack."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        service         = params.get("service", "")
        username        = params.get("username", "")
        username_file   = params.get("username_file", "")
        password        = params.get("password", "")
        password_file   = params.get("password_file", "")
        additional_args = params.get("additional_args", "")

        if not target or not service:
            return jsonify({"error": "Target and service parameters are required"}), 400
        if not (username or username_file) or not (password or password_file):
            return jsonify({"error": "Username/username_file and password/password_file are required"}), 400

        command = "hydra -t 4"
        command += f" -l {username}" if username else f" -L {username_file}"
        command += f" -p {password}" if password else f" -P {password_file}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {target} {service}"

        job = enqueue(f"hydra {service}@{target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/john", methods=["POST"])
def john():
    """Enqueue a John the Ripper cracking job."""
    try:
        params = request.json or {}
        hash_file       = params.get("hash_file", "")
        wordlist        = params.get("wordlist", "/usr/share/wordlists/rockyou.txt")
        format_type     = params.get("format", "")
        additional_args = params.get("additional_args", "")

        if not hash_file:
            return jsonify({"error": "Hash file parameter is required"}), 400

        command = "john"
        if format_type:
            command += f" --format={format_type}"
        if wordlist:
            command += f" --wordlist={wordlist}"
        if additional_args:
            command += f" {additional_args}"
        command += f" {hash_file}"

        job = enqueue(f"john {hash_file}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/wpscan", methods=["POST"])
def wpscan():
    """Enqueue a WPScan scan."""
    try:
        params = request.json or {}
        url             = params.get("url", "")
        additional_args = params.get("additional_args", "")

        if not url:
            return jsonify({"error": "URL parameter is required"}), 400

        command = f"wpscan --url {url}"
        if additional_args:
            command += f" {additional_args}"

        job = enqueue(f"wpscan {url}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/api/tools/enum4linux", methods=["POST"])
def enum4linux():
    """Enqueue an enum4linux scan."""
    try:
        params = request.json or {}
        target          = params.get("target", "")
        additional_args = params.get("additional_args", "-a")

        if not target:
            return jsonify({"error": "Target parameter is required"}), 400

        command = f"enum4linux {additional_args} {target}"

        job = enqueue(f"enum4linux {target}", lambda cmd=command: run_command(cmd))
        return accepted(job)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {e}"}), 500


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Health check — also reports queue depth."""
    essential_tools = ["nmap", "gobuster", "dirb", "nikto"]
    tools_status = {}
    for tool in essential_tools:
        try:
            result = run_command(f"which {tool}")
            tools_status[tool] = result["success"]
        except Exception:
            tools_status[tool] = False

    return jsonify({
        "status":                       "healthy",
        "message":                      "Kali Linux Tools API Server is running",
        "tools_status":                 tools_status,
        "all_essential_tools_available": all(tools_status.values()),
        "queue_depth":                  _job_queue.qsize(),
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Kali Linux API Server")
    parser.add_argument("--debug",  action="store_true", help="Enable debug mode")
    parser.add_argument("--port",   type=int, default=API_PORT,    help=f"API port (default: {API_PORT})")
    parser.add_argument("--ip",     type=str, default="127.0.0.1", help="Bind IP (default: 127.0.0.1)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        DEBUG_MODE = True
        os.environ["DEBUG_MODE"] = "1"
        logger.setLevel(logging.DEBUG)

    if args.port != API_PORT:
        API_PORT = args.port

    logger.info(f"Starting Kali Linux Tools API Server on {args.ip}:{API_PORT}")
    app.run(host=args.ip, port=API_PORT, debug=DEBUG_MODE)
