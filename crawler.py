from collections import deque
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse, urlunparse, urldefrag
from urllib.request import Request, urlopen


SKIPPED_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mp3",
    ".mp4",
    ".ods",
    ".odt",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".svg",
    ".tar",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.links.append(value.strip())


def normalize_url(url, base_url=None, keep_query=False):
    absolute_url = urljoin(base_url, url) if base_url else url
    absolute_url, _fragment = urldefrag(absolute_url)
    parsed = urlparse(absolute_url)

    if parsed.scheme.lower() not in ("http", "https"):
        return None
    if not parsed.netloc:
        return None

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    query = parsed.query if keep_query else ""

    return urlunparse((scheme, netloc, path, "", query, ""))


def extract_links(html):
    parser = LinkExtractor()
    parser.feed(html)
    return parser.links


def is_probably_page_url(url):
    path = urlparse(url).path.lower()
    return not any(path.endswith(extension) for extension in SKIPPED_EXTENSIONS)


def fetch_html(url, timeout_seconds=10):
    request = Request(
        url,
        headers={
            "User-Agent": (
                "UrFU-Diplom-WebGraphCrawler/1.0 "
                "(educational research crawler)"
            )
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            return None

        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def crawl_site_pages(
    start_url,
    max_pages,
    same_domain_only=True,
    timeout_seconds=10,
    max_depth=None,
    keep_query=False,
    fetch_html_func=None,
):
    if max_pages <= 0:
        raise ValueError("max_pages должно быть положительным числом.")

    normalized_start = normalize_url(start_url, keep_query=keep_query)
    if not normalized_start:
        raise ValueError("Стартовый URL должен быть абсолютной http/https-ссылкой.")

    start_host = urlparse(normalized_start).netloc
    fetcher = fetch_html_func or fetch_html
    pages = []
    seen = {normalized_start}
    queue = deque([(normalized_start, 0)])

    while queue and len(pages) < max_pages:
        current_url, depth = queue.popleft()
        if not is_probably_page_url(current_url):
            continue

        try:
            html = fetcher(current_url, timeout_seconds=timeout_seconds)
        except Exception:
            continue
        if html is None:
            continue

        pages.append(current_url)
        if len(pages) >= max_pages:
            break
        if max_depth is not None and depth >= max_depth:
            continue

        for raw_link in extract_links(html):
            next_url = normalize_url(
                raw_link,
                base_url=current_url,
                keep_query=keep_query,
            )
            if not next_url or next_url in seen:
                continue
            if same_domain_only and urlparse(next_url).netloc != start_host:
                continue
            if not is_probably_page_url(next_url):
                continue

            seen.add(next_url)
            queue.append((next_url, depth + 1))

    return pages


def save_page_list(page_paths, output_path):
    with open(output_path, "w", encoding="utf-8") as page_file:
        page_file.write("\n".join(page_paths))
        if page_paths:
            page_file.write("\n")
    return output_path
