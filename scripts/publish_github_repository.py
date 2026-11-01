"""Create/reuse the project repository using existing Git Credential Manager auth.

Credentials stay in process memory and are never written or printed.
"""
import json
import argparse
import subprocess
import urllib.error
import urllib.request

OWNER = 'vgprakul'
NAME = 'hospital-readmission-prediction'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--public', action='store_true', help='Publish this project source repository publicly')
    args = parser.parse_args()
    result = subprocess.run(['git', 'credential', 'fill'],
                            input=f'protocol=https\nhost=github.com\nusername={OWNER}\n\n',
                            text=True, capture_output=True, check=True)
    credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    token = credential.get('password')
    if not token:
        raise RuntimeError('No GitHub credential is available')

    def request(path, payload=None, method=None):
        req = urllib.request.Request('https://api.github.com' + path,
              data=json.dumps(payload).encode() if payload is not None else None,
              headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                       'Content-Type': 'application/json', 'User-Agent': 'hospital-readmission-deployment',
                       'X-GitHub-Api-Version': '2022-11-28'}, method=method)
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)

    try:
        repo = request(f'/repos/{OWNER}/{NAME}')
        action = 'existing'
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise RuntimeError(f'GitHub repository lookup failed: HTTP {error.code}') from None
        try:
            repo = request('/user/repos', {'name': NAME, 'private': True, 'auto_init': False,
                     'description': 'Explainable educational 30-day readmission prediction with a frozen Streamlit pipeline.'})
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'GitHub repository creation failed: HTTP {error.code}') from None
        action = 'created'
    if args.public and repo['private']:
        repo = request(f'/repos/{OWNER}/{NAME}', {'private': False}, method='PATCH')
        action = 'published publicly'
    print(json.dumps({'action': action, 'full_name': repo['full_name'], 'html_url': repo['html_url'],
                      'clone_url': repo['clone_url'], 'private': repo['private'],
                      'default_branch': repo['default_branch'], 'size': repo['size']}, indent=2))


if __name__ == '__main__':
    main()
