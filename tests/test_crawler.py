import os
import tempfile
import unittest

from crawler import crawl_site_pages, normalize_url, save_page_list


class CrawlerTest(unittest.TestCase):
    def test_normalize_url_resolves_relative_links(self):
        self.assertEqual(
            normalize_url("../about/#team", base_url="https://example.test/ru/main/"),
            "https://example.test/ru/about/",
        )

    def test_crawl_site_pages_collects_real_page_urls(self):
        html_by_url = {
            "https://example.test/": """
                <a href="/about">About</a>
                <a href="/docs/report.pdf">PDF</a>
                <a href="mailto:test@example.test">Mail</a>
                <a href="https://external.test/page">External</a>
            """,
            "https://example.test/about": """
                <a href="/contacts#top">Contacts</a>
                <a href="/">Home</a>
            """,
            "https://example.test/contacts": "",
        }

        def fake_fetch(url, timeout_seconds=10):
            return html_by_url[url]

        pages = crawl_site_pages(
            "https://example.test/",
            max_pages=3,
            fetch_html_func=fake_fetch,
        )

        self.assertEqual(
            pages,
            [
                "https://example.test/",
                "https://example.test/about",
                "https://example.test/contacts",
            ],
        )

    def test_crawl_site_pages_can_stop_by_depth(self):
        html_by_url = {
            "https://example.test/": '<a href="/level-1">Level 1</a>',
            "https://example.test/level-1": '<a href="/level-2">Level 2</a>',
        }

        def fake_fetch(url, timeout_seconds=10):
            return html_by_url[url]

        pages = crawl_site_pages(
            "https://example.test/",
            max_pages=5,
            max_depth=0,
            fetch_html_func=fake_fetch,
        )

        self.assertEqual(pages, ["https://example.test/"])

    def test_crawl_site_pages_skips_unavailable_pages(self):
        html_by_url = {
            "https://example.test/": '<a href="/broken">Broken</a><a href="/ok">OK</a>',
            "https://example.test/ok": "",
        }

        def fake_fetch(url, timeout_seconds=10):
            if url not in html_by_url:
                raise OSError("page is unavailable")
            return html_by_url[url]

        pages = crawl_site_pages(
            "https://example.test/",
            max_pages=3,
            fetch_html_func=fake_fetch,
        )

        self.assertEqual(
            pages,
            [
                "https://example.test/",
                "https://example.test/ok",
            ],
        )

    def test_save_page_list_writes_urls(self):
        pages = ["https://example.test/", "https://example.test/about"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "pages.txt")

            saved_path = save_page_list(pages, output_path)

            self.assertEqual(saved_path, output_path)
            with open(output_path, "r", encoding="utf-8") as page_file:
                self.assertEqual(page_file.read(), "\n".join(pages) + "\n")


if __name__ == "__main__":
    unittest.main()
