"""Check local page, fragment, and asset references in the built static site."""
from html.parser import HTMLParser
import os
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1] / "dist"
BASE = "/" + os.environ.get("DOCS_BASE", "/mlzo").strip("/")
BASE = BASE.rstrip("/") + "/"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(path.read_text())

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])


def main():
    pages = {path: Page(path) for path in ROOT.rglob("*.html")}
    if not pages:
        raise SystemExit("No built pages found. Run npm run build first.")
    errors = []
    count = 0
    for path, page in pages.items():
        relative = path.relative_to(ROOT).as_posix()
        route = relative[:-10] if relative.endswith("index.html") else relative
        origin = "https://docs.invalid" + BASE + route
        for link in page.links:
            url = urlsplit(urljoin(origin, link))
            if url.netloc != "docs.invalid" or url.scheme not in ("http", "https"):
                continue
            count += 1
            if not url.path.startswith(BASE):
                errors.append(f"{relative}: outside base {link}")
                continue
            target = ROOT / unquote(url.path[len(BASE):])
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append(f"{relative}: missing {link}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f"{relative}: missing fragment {link}")
    if errors:
        raise SystemExit("\n".join(errors))
    if not (ROOT / "pagefind/pagefind.js").exists():
        raise SystemExit("Search index is missing.")
    print(f"Checked {len(pages)} HTML pages and {count} internal page, fragment, and asset links; search index exists.")


if __name__ == "__main__":
    main()
