#! /usr/bin/python3

# FastMCP client for the Kali Linux Tools API Server (kali_server.py).
#
# Runs locally on the dev workstation and forwards tool calls over HTTP to the
# Kali API server. Every tool endpoint on the backend is async: it returns a
# job_id immediately and the actual scan output is retrieved by polling
# `/api/jobs/<job_id>`. Use `get_job_status` after any tool call to poll.
#
# some of the code here was inspired from https://github.com/whit3rabbit0/project_astro , be sure to check them out

import argparse
import logging
import sys
from typing import Any, Dict, Optional

import requests
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_KALI_SERVER = "https://kali.cstar.diy"   # change to your linux IP
DEFAULT_REQUEST_TIMEOUT = 300                    # 5 minutes


class KaliToolsClient:
    """Client for communicating with the Kali Linux Tools API Server."""

    def __init__(self, server_url: str, timeout: int = DEFAULT_REQUEST_TIMEOUT):
        """
        Initialize the Kali Tools Client.

        Args:
            server_url: URL of the Kali Tools API Server.
            timeout: Request timeout in seconds.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        # logger.info(f"Initialized Kali Tools Client connecting to {server_url}")

    def safe_get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform a GET request with optional query parameters."""
        if params is None:
            params = {}
        url = f"{self.server_url}/{endpoint}"
        try:
            logger.debug(f"GET {url} with params: {params}")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": f"Request failed: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "success": False}

    def safe_post(self, endpoint: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a POST request with JSON data."""
        url = f"{self.server_url}/{endpoint}"
        try:
            logger.debug(f"POST {url} with data: {json_data}")
            response = requests.post(url, json=json_data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return {"error": f"Request failed: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}", "success": False}

    def execute_command(self, command: str) -> Dict[str, Any]:
        """Submit an arbitrary shell command to the Kali server."""
        return self.safe_post("api/command", {"command": command})

    def check_health(self) -> Dict[str, Any]:
        """Check the health of the Kali Tools API Server."""
        return self.safe_get("health")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get the status / result of a previously submitted async job."""
        return self.safe_get(f"api/jobs/{job_id}")


def setup_mcp_server(kali_client: KaliToolsClient) -> FastMCP:
    """Set up the MCP server with all tool functions."""
    mcp = FastMCP("kali_mcp")

    # =========================================================
    # Job polling / server health
    # =========================================================

    @mcp.tool(name="get_job_status")
    def get_job_status(job_id: str) -> Dict[str, Any]:
        """
        Poll the status of a previously submitted async job.

        ALL tool endpoints on this server are async — they return immediately
        with a job_id and status='pending'. Use this to check on the job
        until status is 'completed' or 'failed', at which point 'result'
        (or 'error') will be populated.

        Args:
            job_id: The job_id returned from a previous tool call.

        Returns:
            Job record with status, started_at, finished_at, result, error.
        """
        return kali_client.get_job(job_id)

    @mcp.tool(name="server_health")
    def server_health() -> Dict[str, Any]:
        """
        Check health of the Kali API server.
        Also reports which pentest binaries are installed and the current
        job queue depth.
        """
        return kali_client.check_health()

    # =========================================================
    # Original tool endpoints
    # =========================================================

    @mcp.tool(name="execute_command")
    def execute_command(command: str) -> Dict[str, Any]:
        """
        Execute an arbitrary shell command on the Kali server.
        Use only when no dedicated tool fits. Returns a job_id — poll with
        get_job_status.

        Args:
            command: Shell command to execute on the Kali host.
        """
        return kali_client.execute_command(command)

    @mcp.tool(name="nmap_scan")
    def nmap_scan(
        target: str,
        scan_type: str = "-sCV",
        ports: str = "",
        additional_args: str = "-T4 -Pn",
    ) -> Dict[str, Any]:
        """
        Execute an Nmap scan against a target.

        Args:
            target: IP address or hostname to scan.
            scan_type: Scan flags e.g. -sV, -sC, -A, --script vuln.
            ports: Comma-separated ports or ranges e.g. 80,443 or 1-1000.
            additional_args: Additional Nmap arguments e.g. -T4 -Pn.
        """
        data = {
            "target": target,
            "scan_type": scan_type,
            "ports": ports,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/nmap", data)

    @mcp.tool(name="gobuster_scan")
    def gobuster_scan(
        url: str,
        mode: str = "dir",
        wordlist: str = "/usr/share/wordlists/dirb/common.txt",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Directory/DNS/vhost brute-forcing with Gobuster.

        Args:
            url: Target URL.
            mode: Scan mode — one of dir, dns, fuzz, vhost.
            wordlist: Path to wordlist file.
            additional_args: Additional Gobuster arguments.
        """
        data = {
            "url": url,
            "mode": mode,
            "wordlist": wordlist,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/gobuster", data)

    @mcp.tool(name="dirb_scan")
    def dirb_scan(
        url: str,
        wordlist: str = "/usr/share/wordlists/dirb/common.txt",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Dirb web content scanner.

        Args:
            url: Target URL.
            wordlist: Path to wordlist file.
            additional_args: Additional Dirb arguments.
        """
        data = {
            "url": url,
            "wordlist": wordlist,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/dirb", data)

    @mcp.tool(name="nikto_scan")
    def nikto_scan(target: str, additional_args: str = "") -> Dict[str, Any]:
        """
        Nikto web server vulnerability scanner.

        Args:
            target: Target URL or IP.
            additional_args: Additional Nikto arguments.
        """
        data = {"target": target, "additional_args": additional_args}
        return kali_client.safe_post("api/tools/nikto", data)

    @mcp.tool(name="sqlmap_scan")
    def sqlmap_scan(
        url: str,
        data: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        SQLmap SQL injection scanner.

        Args:
            url: Target URL.
            data: POST data string for form-based injection.
            additional_args: Additional SQLmap arguments.
        """
        post_data = {
            "url": url,
            "data": data,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/sqlmap", post_data)

    @mcp.tool(name="metasploit_run")
    def metasploit_run(
        module: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a Metasploit module via msfconsole resource script. TIER 3 —
        use only for tightly scoped proof-of-concept validation.

        Args:
            module: Module path e.g. exploit/multi/handler,
                    auxiliary/scanner/ssh/ssh_login.
            options: Dict of module options e.g. {"RHOSTS": "10.0.0.1",
                    "LPORT": "4444"}.
        """
        data = {"module": module, "options": options or {}}
        return kali_client.safe_post("api/tools/metasploit", data)

    @mcp.tool(name="hydra_attack")
    def hydra_attack(
        target: str,
        service: str,
        username: str = "",
        username_file: str = "",
        password: str = "",
        password_file: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Hydra credential testing. TIER 3 — high risk; only when authorized for
        the engagement.

        Args:
            target: Target IP or hostname.
            service: Service to attack e.g. ssh, ftp, http-post-form, smb.
            username: Single username to try.
            username_file: Path to username list.
            password: Single password to try.
            password_file: Path to password list.
            additional_args: Additional Hydra arguments.
        """
        data = {
            "target": target,
            "service": service,
            "username": username,
            "username_file": username_file,
            "password": password,
            "password_file": password_file,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/hydra", data)

    @mcp.tool(name="john_crack")
    def john_crack(
        hash_file: str,
        wordlist: str = "/usr/share/wordlists/rockyou.txt",
        format_type: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        John the Ripper password cracker. TIER 3 — only on hashes you are
        authorized to analyze.

        Args:
            hash_file: Path to hash file on the Kali server.
            wordlist: Path to wordlist file.
            format_type: Hash format e.g. nt, md5crypt, sha512crypt, raw-sha256.
            additional_args: Additional John arguments.
        """
        data = {
            "hash_file": hash_file,
            "wordlist": wordlist,
            "format": format_type,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/john", data)

    @mcp.tool(name="wpscan_analyze")
    def wpscan_analyze(url: str, additional_args: str = "") -> Dict[str, Any]:
        """
        WPScan WordPress vulnerability scanner.

        Args:
            url: Target WordPress site URL.
            additional_args: Additional WPScan arguments e.g. --enumerate u,p,vp.
        """
        data = {"url": url, "additional_args": additional_args}
        return kali_client.safe_post("api/tools/wpscan", data)

    @mcp.tool(name="enum4linux_scan")
    def enum4linux_scan(target: str, additional_args: str = "-a") -> Dict[str, Any]:
        """
        Enum4linux Windows/Samba SMB enumeration tool.

        Args:
            target: Target IP or hostname.
            additional_args: Additional enum4linux arguments (default -a for
                    all checks).
        """
        data = {"target": target, "additional_args": additional_args}
        return kali_client.safe_post("api/tools/enum4linux", data)

    # =========================================================
    # Red Team endpoints
    # =========================================================

    @mcp.tool(name="theharvester_scan")
    def theharvester_scan(
        domain: str,
        sources: str = "baidu,bing,crtsh,dnsdumpster,duckduckgo,hackertarget,rapiddns,urlscan",
        limit: int = 500,
        dns_resolve: bool = False,
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        theHarvester passive OSINT. TIER 1 — harvest emails, subdomains, hosts,
        and employee names from public sources.

        Args:
            domain: Target domain to harvest.
            sources: Comma-separated data sources.
            limit: Max results to retrieve.
            dns_resolve: Resolve discovered hosts to IPs.
            additional_args: Additional theHarvester arguments.
        """
        data = {
            "domain": domain,
            "sources": sources,
            "limit": limit,
            "dns_resolve": dns_resolve,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/theharvester", data)

    @mcp.tool(name="amass_enum")
    def amass_enum(
        domain: str,
        mode: str = "passive",
        wordlist: str = "",
        output_dir: str = "/tmp/amass_out",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Amass subdomain enumeration. TIER 1 (passive) / TIER 2 (active/brute).

        Args:
            domain: Target domain.
            mode: passive | active | brute.
            wordlist: Path to wordlist — required when mode='brute'.
            output_dir: Directory for Amass output files.
            additional_args: Additional Amass arguments.
        """
        data = {
            "domain": domain,
            "mode": mode,
            "wordlist": wordlist,
            "output_dir": output_dir,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/amass", data)

    @mcp.tool(name="recon_ng_run")
    def recon_ng_run(
        module: str,
        source: str,
        workspace: str = "default",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Recon-ng modular OSINT framework. TIER 1.

        Args:
            module: Full module path e.g. recon/domains-hosts/hackertarget.
            source: Source value (domain, IP, company name, etc.).
            workspace: Recon-ng workspace name.
            additional_args: Extra key=value options (comma-separated) e.g.
                    "LIMIT=10,VERBOSE=true".
        """
        data = {
            "module": module,
            "source": source,
            "workspace": workspace,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/recon-ng", data)

    @mcp.tool(name="netexec_run")
    def netexec_run(
        target: str,
        protocol: str,
        username: str = "",
        username_file: str = "",
        password: str = "",
        password_file: str = "",
        ntlm_hash: str = "",
        action: str = "",
        domain: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        NetExec (nxc) SMB/WinRM/LDAP/RDP/SSH testing. TIER 2-3.

        Args:
            target: IP, CIDR, or hostname.
            protocol: smb | winrm | ldap | rdp | ssh | mssql.
            username: Single username.
            username_file: Path to username list.
            password: Single password.
            password_file: Path to password list.
            ntlm_hash: NTLM hash for Pass-the-Hash (LMHASH:NTHASH or just
                    NTHASH).
            action: Module/action flag e.g. --shares, --users, --sam, --lsa.
            domain: Windows domain name.
            additional_args: Additional netexec arguments.
        """
        data = {
            "target": target,
            "protocol": protocol,
            "username": username,
            "username_file": username_file,
            "password": password,
            "password_file": password_file,
            "hash": ntlm_hash,
            "action": action,
            "domain": domain,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/netexec", data)

    @mcp.tool(name="impacket_run")
    def impacket_run(
        script: str,
        target: str,
        username: str = "",
        password: str = "",
        ntlm_hash: str = "",
        domain: str = "",
        dc_ip: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Impacket script execution. TIER 3 — Windows/AD protocol abuse.
        Use only on explicitly authorized targets.

        Args:
            script: secretsdump | GetNPUsers | GetUserSPNs | psexec | wmiexec |
                    smbclient | ticketer | getPac.
            target: Target IP or hostname.
            username: Username.
            password: Password.
            ntlm_hash: NTLM hash (LMHASH:NTHASH or just NTHASH).
            domain: Domain name.
            dc_ip: Domain controller IP.
            additional_args: Extra arguments for the specific script.
        """
        data = {
            "script": script,
            "target": target,
            "username": username,
            "password": password,
            "hash": ntlm_hash,
            "domain": domain,
            "dc_ip": dc_ip,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/impacket", data)

    @mcp.tool(name="responder_run")
    def responder_run(
        interface: str,
        duration_seconds: int = 60,
        modes: str = "--wpad",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Responder LLMNR/NBT-NS/mDNS poisoning. TIER 3 — captures NTLMv2 hashes
        on internal segments. Use only on authorized internal networks.

        Args:
            interface: Network interface to listen on e.g. eth0.
            duration_seconds: How long to run before auto-stopping (1-3600).
            modes: Extra mode flags e.g. --wpad --lm.
            additional_args: Additional Responder arguments.
        """
        data = {
            "interface": interface,
            "duration_seconds": duration_seconds,
            "modes": modes,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/responder", data)

    @mcp.tool(name="evil_winrm_run")
    def evil_winrm_run(
        target: str,
        username: str,
        command: str,
        password: str = "",
        ntlm_hash: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Evil-WinRM single-command execution. TIER 3 — WinRM post-exploitation
        shell access.

        Args:
            target: Target IP or hostname.
            username: Username.
            command: Command to execute on the remote host.
            password: Password (use either password OR hash).
            ntlm_hash: NTLM hash for Pass-the-Hash.
            additional_args: Additional Evil-WinRM arguments.
        """
        data = {
            "target": target,
            "username": username,
            "command": command,
            "password": password,
            "hash": ntlm_hash,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/evil-winrm", data)

    @mcp.tool(name="bloodhound_collect")
    def bloodhound_collect(
        domain: str,
        dc_ip: str,
        username: str,
        password: str = "",
        ntlm_hash: str = "",
        collection_methods: str = "Default",
        output_dir: str = "/tmp/bh_out",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        BloodHound data collection (bloodhound-python). TIER 2 — collects AD
        data for attack-path analysis. Visualization happens in the BloodHound
        UI separately.

        Args:
            domain: Target domain.
            dc_ip: Domain controller IP.
            username: Domain username.
            password: Password.
            ntlm_hash: NTLM hash (LMHASH:NTHASH or just NTHASH).
            collection_methods: Default | All | DCOnly | Session | Trusts |
                    ACL | Group.
            output_dir: Directory for JSON output files.
            additional_args: Additional bloodhound-python arguments.
        """
        data = {
            "domain": domain,
            "dc_ip": dc_ip,
            "username": username,
            "password": password,
            "hash": ntlm_hash,
            "collection_methods": collection_methods,
            "output_dir": output_dir,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/bloodhound", data)

    @mcp.tool(name="aircrack_run")
    def aircrack_run(
        action: str,
        interface: str,
        bssid: str = "",
        channel: int = 0,
        capture_file: str = "/tmp/capture",
        wordlist: str = "/usr/share/wordlists/rockyou.txt",
        duration_seconds: int = 60,
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Aircrack-ng Wi-Fi assessment. TIER 3 — only on networks you are
        explicitly authorized to test.

        Args:
            action: monitor_start | capture | deauth | crack.
            interface: Wireless interface e.g. wlan0.
            bssid: Target AP BSSID — required for capture/deauth.
            channel: Channel to lock on (0 = auto, used for capture).
            capture_file: Output/input .cap file path.
            wordlist: Wordlist path — used for crack.
            duration_seconds: Capture duration in seconds.
            additional_args: Additional arguments.
        """
        data = {
            "action": action,
            "interface": interface,
            "bssid": bssid,
            "channel": channel,
            "capture_file": capture_file,
            "wordlist": wordlist,
            "duration_seconds": duration_seconds,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/aircrack", data)

    @mcp.tool(name="bettercap_run")
    def bettercap_run(
        interface: str,
        caplet: str = "",
        commands: str = "",
        duration_seconds: int = 60,
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Bettercap network attacks/recon. TIER 3 — MITM, ARP spoofing, Wi-Fi,
        BLE. Use only on authorized network segments.

        Args:
            interface: Network interface e.g. eth0.
            caplet: Path to a .cap caplet script file (either caplet or
                    commands is required).
            commands: Semicolon-separated bettercap commands to evaluate.
            duration_seconds: Run duration before auto-stop (1-3600).
            additional_args: Additional bettercap arguments.
        """
        data = {
            "interface": interface,
            "caplet": caplet,
            "commands": commands,
            "duration_seconds": duration_seconds,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/bettercap", data)

    # =========================================================
    # Blue Team endpoints
    # =========================================================

    @mcp.tool(name="volatility_run")
    def volatility_run(
        image: str,
        plugin: str,
        pid: int = 0,
        output_format: str = "text",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Volatility 3 memory forensics. TIER 1 — analyze a memory dump for
        processes, network connections, injected code, credentials, etc.

        Args:
            image: Path to memory dump file on the Kali server.
            plugin: Plugin to run e.g. windows.pslist, windows.netscan,
                    windows.malfind, windows.hashdump, windows.cmdline,
                    windows.dlllist, windows.dumpfiles, linux.pslist,
                    linux.bash.
            pid: Filter by PID (0 = all).
            output_format: text | json.
            additional_args: Additional Volatility arguments.
        """
        data = {
            "image": image,
            "plugin": plugin,
            "pid": pid,
            "output_format": output_format,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/volatility", data)

    @mcp.tool(name="lynis_audit")
    def lynis_audit(
        target: str = "localhost",
        profile: str = "default",
        categories: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Lynis Linux hardening audit. TIER 1 — blue team CIS/baseline check.

        Args:
            target: 'localhost' or remote SSH target user@host.
            profile: Lynis profile name.
            categories: Comma-separated test categories to run.
            additional_args: Additional Lynis arguments.
        """
        data = {
            "target": target,
            "profile": profile,
            "categories": categories,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/lynis", data)

    @mcp.tool(name="trivy_scan")
    def trivy_scan(
        target: str,
        scan_type: str = "image",
        severity: str = "HIGH,CRITICAL",
        ignore_unfixed: bool = False,
        output_format: str = "table",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Trivy vulnerability and misconfiguration scanner. TIER 1.

        Args:
            target: Image name, directory path, git URL, or IaC path.
            scan_type: image | fs | repo | config | sbom.
            severity: Severity filter e.g. LOW,MEDIUM,HIGH,CRITICAL.
            ignore_unfixed: Skip unfixed vulnerabilities.
            output_format: table | json | sarif.
            additional_args: Additional Trivy arguments.
        """
        data = {
            "target": target,
            "scan_type": scan_type,
            "severity": severity,
            "ignore_unfixed": ignore_unfixed,
            "output_format": output_format,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/trivy", data)

    @mcp.tool(name="semgrep_scan")
    def semgrep_scan(
        path: str,
        config: str = "auto",
        severity: str = "",
        output_format: str = "text",
        exclude: str = "",
        additional_args: str = "",
    ) -> Dict[str, Any]:
        """
        Semgrep SAST scanner. TIER 1 — static analysis for C#, TypeScript,
        Python, etc.

        Args:
            path: Path to source directory or file on the Kali server.
            config: Ruleset — auto | p/csharp | p/typescript | p/owasp-top-ten |
                    p/secrets | path to custom rules.
            severity: Filter — ERROR | WARNING | INFO.
            output_format: text | json | sarif.
            exclude: Comma-separated glob patterns to exclude.
            additional_args: Additional Semgrep arguments.
        """
        data = {
            "path": path,
            "config": config,
            "severity": severity,
            "output_format": output_format,
            "exclude": exclude,
            "additional_args": additional_args,
        }
        return kali_client.safe_post("api/tools/semgrep", data)

    return mcp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Kali MCP Client")
    parser.add_argument(
        "--server",
        type=str,
        default=DEFAULT_KALI_SERVER,
        help=f"Kali API server URL (default: {DEFAULT_KALI_SERVER})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main():
    """Main entry point for the MCP server."""
    args = parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Initialize the Kali Tools client
    kali_client = KaliToolsClient(args.server, args.timeout)

    # Check server health and log the result
    health = kali_client.check_health()
    if "error" in health:
        logger.warning(f"Unable to connect to Kali API server at {args.server}: {health['error']}")
        logger.warning("MCP server will start, but tool execution may fail")
    else:
        # logger.info(f"Successfully connected to Kali API server at {args.server}")
        # logger.info(f"Server health status: {health['status']}")
        if not health.get("all_essential_tools_available", False):
            logger.warning("Not all essential tools are available on the Kali server")
            missing_tools = [
                tool for tool, available in health.get("tools_status", {}).items()
                if not available
            ]
            if missing_tools:
                logger.warning(f"Missing tools: {', '.join(missing_tools)}")

    # Set up and run the MCP server
    mcp = setup_mcp_server(kali_client)
    # logger.info("Starting Kali MCP server")
    mcp.run()


if __name__ == "__main__":
    main()