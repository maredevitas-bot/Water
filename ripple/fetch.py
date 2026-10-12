"""Download official annual ZIPs; retain their hashes and source-page links."""
import re
import time
import urllib.request
import zipfile
from urllib.parse import urljoin

from .common import ROOT, dump_json, read_config, sha256, timestamp

PAGE = 'https://www.eia.gov/electricity/data/eia923/'


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'RippleEffectResearch/0.1'})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    raw = ROOT / 'data' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    html = get(PAGE).decode('utf-8')
    links = [urljoin(PAGE, x) for x in re.findall(r'href=["\']([^"\']+\.zip)', html)]
    manifest_path = raw / 'manifest.json'
    import json
    old = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    by_name = {x['file']: x for x in old}
    for year in read_config()['years']:
        candidates = [x for x in links if x.endswith(f'/f923_{year}.zip')]
        if len(set(candidates)) != 1:
            raise ValueError(f'Expected one official ZIP link for {year}: {candidates}')
        url = candidates[0]
        dest = raw / f'f923_{year}.zip'
        downloaded = not dest.exists()
        if downloaded:
            print(f'Downloading {year}...', flush=True)
            temp = dest.with_suffix('.zip.part')
            temp.write_bytes(get(url))
            if not zipfile.is_zipfile(temp):
                raise ValueError(f'Not a ZIP: {url}; original response retained as {temp}')
            with zipfile.ZipFile(temp) as z:
                if z.testzip() is not None:
                    raise ValueError(f'ZIP integrity error: {url}')
            temp.replace(dest)
        digest = sha256(dest)
        previous = by_name.get(dest.name)
        if previous and previous['sha256'] != digest:
            raise ValueError(f'Cached raw file hash changed: {dest.name}')
        with zipfile.ZipFile(dest) as z:
            members = z.namelist()
            if not any('Schedule_8_' in x for x in members):
                raise ValueError(f'Schedule 8 missing in {dest.name}')
        by_name[dest.name] = {
            'file': dest.name, 'year': year, 'url': url, 'source_page': PAGE,
            'sha256': digest, 'bytes': dest.stat().st_size,
            'downloaded_at_utc': timestamp() if downloaded else (previous or {}).get('downloaded_at_utc'),
            'verified_at_utc': timestamp(), 'members': members,
            'note': 'Cached file checked locally; not silently refreshed from server.'
        }
        dump_json(manifest_path, list(by_name.values()))
        print(f'Ready: {dest.name}', flush=True)


if __name__ == '__main__':
    main()
