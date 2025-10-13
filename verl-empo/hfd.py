#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
import json
import re
from pathlib import Path
import shutil
import signal

# Color definitions using ANSI escape codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color

def signal_handler(sig, frame):
    print(f"{Colors.YELLOW}\nDownload interrupted. You can resume by re-running the command.{Colors.NC}")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def display_help():
    help_text = """
Usage:
  hfd <REPO_ID> [--include include_pattern1 include_pattern2 ...] [--exclude exclude_pattern1 exclude_pattern2 ...] [--hf_username username] [--hf_token token] [--tool aria2c|wget] [-x threads] [-j jobs] [--dataset] [--local-dir path] [--revision rev]
Description:
  Downloads a model or dataset from Hugging Face using the provided repo ID.
Arguments:
  REPO_ID         The Hugging Face repo ID (Required)
                  Format: 'org_name/repo_name' or legacy format (e.g., gpt2)
Options:
  include/exclude_pattern The patterns to match against file path, supports wildcard characters.
                  e.g., '--exclude *.safetensor *.md', '--include vae/*'.
  --include       (Optional) Patterns to include files for downloading (supports multiple patterns).
  --exclude       (Optional) Patterns to exclude files from downloading (supports multiple patterns).
  --hf_username   (Optional) Hugging Face username for authentication (not email).
  --hf_token      (Optional) Hugging Face token for authentication.
  --tool          (Optional) Download tool to use: aria2c (default) or wget.
  -x              (Optional) Number of download threads for aria2c (default: 3).
  -j              (Optional) Number of concurrent downloads for aria2c (default: 4).
  --dataset       (Optional) Flag to indicate downloading a dataset.
  --local_dir     (Optional) Directory path to store the downloaded data.
  --revision      (Optional) Model/Dataset revision to download (default: main).
  --enable_mirror (Optional) Download from hf-mirror.com.
Example:
  python hfd.py gpt2
  python hfd.py bigscience/bloom-560m --exclude *.safetensors --enable_mirror
  python hfd.py meta-llama/Llama-2-7b --hf_username myuser --hf_token mytoken -x 4
  python hfd.py lavita/medical-qa-shared-task-v1-toy --dataset
  python hfd.py bartowski/Phi-3.5-mini-instruct-exl2 --revision 5_0 --local_dir your_local_path
"""
    print(help_text)
    sys.exit(1)

def validate_number(value, name, max_value):
    if not (value.isdigit() and 1 <= int(value) <= max_value):
        print(f"{Colors.RED}[Error] {name} must be 1-{max_value}{Colors.NC}")
        sys.exit(1)
    return int(value)

def check_command(command):
    if not shutil.which(command):
        print(f"{Colors.RED}{command} is not installed. Please install it first.{Colors.NC}")
        sys.exit(1)

def generate_command_string(args):
    return " ".join([
        f"REPO_ID={args.repo_id}",
        f"TOOL={args.tool}",
        f"INCLUDE_PATTERNS={' '.join(args.include)}",
        f"EXCLUDE_PATTERNS={' '.join(args.exclude)}",
        f"DATASET={1 if args.dataset else 0}",
        f"HF_USERNAME={args.hf_username or ''}",
        f"HF_TOKEN={args.hf_token or ''}",
        f"HF_ENDPOINT={args.hf_endpoint}",
        f"REVISION={args.revision}"
    ])

def fetch_and_save_metadata(args, metadata_file):
    headers = {"Authorization": f"Bearer {args.hf_token}"} if args.hf_token else {}
    cmd = ["curl", "-L", "-s", "-o", str(metadata_file), args.api_url]
    if args.hf_token:
        cmd.extend(["-H", f"Authorization: Bearer {args.hf_token}"])
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"{Colors.RED}[Error] Failed to fetch metadata from {args.api_url}. Error: {process.stderr}{Colors.NC}")
        if metadata_file.exists():
            metadata_file.unlink()
        sys.exit(1)
    
    with open(metadata_file, 'r') as f:
        return f.read()

def check_authentication(response, args):
    try:
        data = json.loads(response)
        gated = data.get('gated', False)
        if gated and (not args.hf_token or not args.hf_username):
            print(f"{Colors.RED}The repository requires authentication, but --hf_username and --hf_token is not passed. "
                  f"Please get token from https://huggingface.co/settings/tokens.\nExiting.{Colors.NC}")
            sys.exit(1)
    except json.JSONDecodeError:
        if '"gated":true' in response and (not args.hf_token or not args.hf_username):
            print(f"{Colors.RED}The repository requires authentication, but --hf_username and --hf_token is not passed. "
                  f"Please get token from https://huggingface.co/settings/tokens.\nExiting.{Colors.NC}")
            sys.exit(1)

def should_regenerate_filelist(args, local_dir):
    command_file = local_dir / '.hfd' / 'last_download_command'
    current_command = generate_command_string(args)
    fileslist_file = local_dir / args.fileslist_file
    
    if not fileslist_file.exists() or not command_file.exists():
        command_file.parent.mkdir(parents=True, exist_ok=True)
        with open(command_file, 'w') as f:
            f.write(current_command)
        return True
    
    with open(command_file, 'r') as f:
        saved_command = f.read().strip()
    
    if current_command != saved_command:
        with open(command_file, 'w') as f:
            f.write(current_command)
        return True
    return False

def generate_file_list(args, response, local_dir):
    fileslist_file = local_dir / args.fileslist_file
    
    include_regex = '|'.join(re.escape(p).replace(r'\*', '.*') for p in args.include) if args.include else ""
    exclude_regex = '|'.join(re.escape(p).replace(r'\*', '.*') for p in args.exclude) if args.exclude else ""
    
    try:
        data = json.loads(response)
        files = [sib['rfilename'] for sib in data.get('siblings', []) if sib.get('rfilename')]
        
        filtered_files = []
        for file in files:
            if (not include_regex or re.search(include_regex, file)) and \
                (not exclude_regex or not re.search(exclude_regex, file)):
                filtered_files.append(file)
        
        with open(fileslist_file, 'w') as f:
            for file in filtered_files:
                url = f"{args.hf_endpoint}/{args.download_api_path}/resolve/{args.revision}/{file}"
                if args.tool == "aria2c":
                    f.write(f"{url}\n")
                    f.write(f"  dir={os.path.dirname(file)}\n")
                    f.write(f"  out={os.path.basename(file)}\n")
                    if args.hf_token:
                        f.write(f"  header=Authorization: Bearer {args.hf_token}\n")
                    f.write("\n")
                else:
                    f.write(f"{url}\n")
    
    except json.JSONDecodeError:
        print(f"{Colors.YELLOW}[Warning] Failed to parse JSON, using basic filtering{Colors.NC}")
        files = re.findall(r'"rfilename":"([^"]*)"', response)
        
        with open(fileslist_file, 'w') as f:
            for file in files:
                if (not include_regex or re.search(include_regex, file)) and \
                   (not exclude_regex or not re.search(exclude_regex, file)):
                    url = f"{args.hf_endpoint}/{args.download_api_path}/resolve/{args.revision}/{file}"
                    if args.tool == "aria2c":
                        f.write(f"{url}\n")
                        f.write(f"  dir={os.path.dirname(file)}\n")
                        f.write(f"  out={os.path.basename(file)}\n")
                        if args.hf_token:
                            f.write(f"  header=Authorization: Bearer {args.hf_token}\n")
                        f.write("\n")
                    else:
                        f.write(f"{url}\n")

def verify_files(local_dir, metadata):
    data = json.loads(metadata)
    for sib in data.get('siblings', []):
        file_path = local_dir / sib['rfilename']
        expected_size = sib.get('size')
        if expected_size and file_path.exists() and file_path.stat().st_size != expected_size:
            print(f"{Colors.RED}File {file_path} is incomplete!{Colors.NC}")
            return False
    return True

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('repo_id', default='TMLR-Group-HF/CoReward-Qwen3-8B-Base', nargs='?', help='The Hugging Face repo ID')
    parser.add_argument('--include', nargs='*', default=[], help='Patterns to include files')
    parser.add_argument('--exclude', nargs='*', default=[], help='Patterns to exclude files')
    parser.add_argument('--hf_username', default='qingyangzhang', help='Hugging Face username')
    parser.add_argument('--hf_token', default='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', help='Hugging Face token')
    parser.add_argument('--tool', choices=['aria2c', 'wget'], default='aria2c', help='Download tool')
    parser.add_argument('-x', type=lambda x: validate_number(x, 'threads (-x)', 10), default=3, help='Number of threads')
    parser.add_argument('-j', type=lambda x: validate_number(x, 'concurrent downloads (-j)', 10), default=4, help='Number of concurrent downloads')
    parser.add_argument('--dataset', action='store_true', help='Flag to download dataset')
    parser.add_argument('--local_dir', default='/fs-computility/prime/shared/zqy_models/checkpoints/hf_models/CoReward-Qwen3-8B-Base', help='Directory to store downloaded data')
    parser.add_argument('--revision', default='main', help='Model/Dataset revision')
    parser.add_argument('--enable_mirror', action='store_true')
    parser.add_argument('--max_retries', type=int, default=1000000)
    
    args = parser.parse_args()
    
    if not args.repo_id or args.repo_id in ('-h', '--help'):
        display_help()
        return
    
    args.hf_endpoint = 'https://hf-mirror.com' if args.enable_mirror else 'https://huggingface.co'
    args.local_dir = Path(args.local_dir if args.local_dir else args.repo_id.split('/')[-1])
    args.local_dir.mkdir(parents=True, exist_ok=True)
    (args.local_dir / '.hfd').mkdir(exist_ok=True)
    
    if args.dataset:
        args.metadata_api_path = f"datasets/{args.repo_id}"
        args.download_api_path = f"datasets/{args.repo_id}"
        args.cut_dirs = 5
    else:
        args.metadata_api_path = f"models/{args.repo_id}"
        args.download_api_path = args.repo_id
        args.cut_dirs = 4
    
    if args.revision != "main":
        args.metadata_api_path = f"{args.metadata_api_path}/revision/{args.revision}"
    args.api_url = f"{args.hf_endpoint}/api/{args.metadata_api_path}"
    args.metadata_file = args.local_dir / '.hfd' / 'repo_metadata.json'
    args.fileslist_file = Path(f".hfd/{args.tool}_urls.txt")
    
    check_command('curl')
    check_command(args.tool)
    
    if not args.metadata_file.exists():
        print(f"{Colors.YELLOW}Fetching repo metadata...{Colors.NC}")
        response = fetch_and_save_metadata(args, args.metadata_file)
        check_authentication(response, args)
    else:
        print(f"{Colors.GREEN}Using cached metadata: {args.metadata_file}{Colors.NC}")
        with open(args.metadata_file, 'r') as f:
            response = f.read()
        check_authentication(response, args)
    
    if should_regenerate_filelist(args, args.local_dir):
        if args.fileslist_file.exists():
            args.fileslist_file.unlink()
        print(f"{Colors.YELLOW}Generating file list...{Colors.NC}")
        generate_file_list(args, response, args.local_dir)
    else:
        print(f"{Colors.GREEN}Resume from file list: {args.local_dir / args.fileslist_file}{Colors.NC}")
    
    print(f"{Colors.YELLOW}Starting download with {args.tool} to {args.local_dir}...{Colors.NC}")
    os.chdir(args.local_dir)
    
    if args.tool == "aria2c":
        cmd = [
            "aria2c", "--summary-interval=1", "--file-allocation=none", 
            "--check-integrity=true","--max-tries=10000000","--retry-wait=1",
            "--disk-cache=32M", "--async-dns=false", "--disable-ipv6=true",
            "-x", str(args.x), "-j", str(args.j), "-s", str(args.x),
            "-k", "1M", "-c", "-i", args.fileslist_file, 
            "--save-session", args.fileslist_file,
        ]
    else:
        cmd = [
            "wget", "-x", "-nH", f"--cut-dirs={args.cut_dirs}",
            "--input-file", args.fileslist_file, "--continue",
            "--progress=bar:force", "nv",  # Changed to refresh progress
        ]
        if args.hf_token:
            cmd.append(f"--header=Authorization: Bearer {args.hf_token}")
        
    # Use Popen instead of run to handle output in real-time
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Print progress with carriage return for refresh effect
    for line in process.stdout:
        if 'redirect' not in line.lower():
            print(f"\r{line.strip()}", flush=True)
    
    process.wait()
    
    print()  # New line after download completes

    max_retries = args.max_retries
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        print(f"{Colors.YELLOW}Attempt {attempt} of {max_retries}...{Colors.NC}")
        
        process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # Print progress
        for line in process.stdout:
            if 'redirect' not in line.lower():
                print(f"\r{line.strip()}", flush=True, end="")
        
        process.wait()
        
        print()  # New line after download attempt
        
        if process.returncode == 0:
            print(f"{Colors.GREEN}Download completed successfully. Repo directory: {os.getcwd()}{Colors.NC}")
            break
        else:
            print(f"{Colors.RED}Download failed with return code {process.returncode}. Retrying...{Colors.NC}")
            if attempt == max_retries:
                print(f"{Colors.RED}Max retries ({max_retries}) reached. Download failed.{Colors.NC}")
                sys.exit(1)

if __name__ == "__main__":
    main()