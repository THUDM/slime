"""Exercise the built bilingual lab in Chromium; no GPU or server required."""

import functools
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from playwright.sync_api import expect, sync_playwright


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Navigation can cancel an in-flight asset request.


def expect_rendered_math(page):
    # Raw TeX is also visible: require MathJax output for every formula, including
    # nodes nested inside MyST's tex2jax_ignore / mathjax_ignore article wrapper.
    expect(page.locator(".reader-article .math").first).to_be_visible()
    expect(page.locator(".reader-article .math:not(:has(mjx-container))")).to_have_count(0, timeout=30000)
    expect(page.locator(".reader-article mjx-merror")).to_have_count(0)


def run():
    build = Path(__file__).resolve().parents[1] / "build"
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(build)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as playwright, tempfile.TemporaryDirectory() as downloads:
            browser = playwright.chromium.launch()
            for lang in ("en", "zh"):
                context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
                context.grant_permissions(["clipboard-read", "clipboard-write"])
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error, target=errors: target.append(str(error)))
                page.goto(f"{base}/{lang}/", wait_until="networkidle")
                expect(page.locator('[name="model"]')).to_have_count(6)
                expect(page.locator(".hero-description")).to_contain_text("slime")
                expect(page.locator(".core-capabilities article")).to_have_count(2)
                expect(page.locator(".glm-lineage")).to_contain_text("GLM-5.3-Flash")
                expect(page.locator('[name="model"][value="glm5"] + span')).to_contain_text("GLM-5.3")
                assert (
                    page.locator(".option-description").first.evaluate(
                        "el => parseFloat(getComputedStyle(el).fontSize)"
                    )
                    >= 16
                )
                expect(page.locator('[data-signal="math"]')).to_be_visible()
                page.locator('[name="task"][value="custom"]').check()
                expect(page.locator('[data-signal="custom"]')).to_be_visible()
                page.locator("#undo").click()
                expect(page.locator('[data-signal="math"]')).to_be_visible()
                page.locator('.library-teaser a[href="get_started/experiment-guide.html"]').click()
                expect(page.locator(".reader-article .mermaid")).to_be_visible()
                assert page.url == f"{base}/{lang}/"
                page.keyboard.press("Escape")
                for guide in ("policy-mismatch", "rl-systems"):
                    page.locator(f'.library-teaser a[href="advanced/{guide}.html"]').click()
                    expect_rendered_math(page)
                    page.keyboard.press("Escape")
                expect(page.locator("#download")).to_be_enabled()
                page.locator('[data-step="1"]').click()
                page.locator('[name="layout"][value="external"]').check()
                expect(page.locator(".external-map .sampler")).to_have_count(3)
                page.locator("#undo").click()
                expect(page.locator(".external-map")).to_have_count(0)
                page.locator('[data-parallel="cp"]').click()
                page.locator('[data-cp="allgather"]').click()
                expect(page.locator("#parallel-diagram")).to_contain_text("DSA")
                # The illustration must not change the selected model's CP recipe.
                assert "--allgather-cp" not in page.locator("#artifact-code").inner_text()
                page.locator('[data-parallel="memory"]').click()
                page.locator('[name="cpuAdam"]').uncheck()
                assert "--optimizer-cpu-offload" not in page.locator("#artifact-code").inner_text()
                page.locator("#undo").click()
                expect(page.locator('[name="cpuAdam"]')).to_be_checked()
                page.locator("#redo").click()
                expect(page.locator('[name="cpuAdam"]')).not_to_be_checked()
                page.locator('[name="partial"]').check()
                page.locator('[name="maskPartial"]').check()
                assert "--over-sampling-batch-size 16" in page.locator("#artifact-code").inner_text()
                page.locator('[name="schedule"][value="async"]').check()
                expect(page.locator('[name="layout"][value="separate"]')).to_be_checked()
                page.locator('[name="straw"]').check()
                expect(page.locator("#download")).to_be_disabled()
                for name, value in {
                    "queueDir": "/shared/juicefs/my-run",
                    "queueRun": "browser-run",
                    "declaration": "/shared/deployment.json",
                }.items():
                    page.locator(f'[name="{name}"]').fill(value)
                    page.locator(f'[name="{name}"]').press("Tab")
                expect(page.locator("#download")).to_be_enabled()
                expect(page.locator(".schedule-explorer")).to_contain_text("Distributed fully async")
                expect(page.locator(".queue-pool")).to_contain_text("straw")
                script = page.locator("#artifact-code").inner_text()
                assert "--rollout-data-transport straw" in script and "--partial-rollout" in script
                assert "--over-sampling-batch-size" not in script
                page.locator('[data-learn="partial"]').click()
                expect(page.locator("#learn-dialog")).to_be_visible()
                expect(page.locator(".reader-article")).to_be_visible()
                expect(page.locator(".reader-sources a")).to_have_count(2)
                assert len(page.locator(".reader-article").inner_text()) > 500
                assert page.url == f"{base}/{lang}/"
                expect_rendered_math(page)
                page.keyboard.press("Escape")
                expect(page.locator('[data-learn="partial"]')).to_be_focused()
                page.reload(wait_until="networkidle")
                expect(page.locator('[name="queueRun"]')).to_have_value("browser-run")
                page.locator("#share").click()
                shared = page.evaluate("navigator.clipboard.readText()")
                assert "browser-run" not in shared and "deployment.json" not in shared
                shared_page = context.new_page()
                shared_page.goto(shared)
                expect(shared_page.locator('[name="straw"]')).to_be_checked()
                expect(shared_page.locator('[name="queueDir"]')).to_have_value("")
                shared_page.close()
                with page.expect_download() as event:
                    page.locator("#download").click()
                target = Path(downloads) / f"{lang}-experiment.sh"
                event.value.save_as(target)
                subprocess.run(["bash", "-n", target], check=True)
                assert target.read_text() == script
                # PD validates both engine pools; YAML disables cache on decode.
                page.locator("#reset").click()
                page.locator('[data-step="1"]').click()
                page.locator('[name="layout"][value="separate"]').check()
                page.locator('[name="rollout"]').fill("16")
                page.locator('[name="rollout"]').press("Tab")
                page.locator('[data-step="4"]').click()
                page.locator('[name="pd"]').check()
                page.locator('[name="hicache"]').check()
                page.locator('[name="ib"]').fill("mlx5_0")
                page.locator('[name="ib"]').press("Tab")
                page.locator('[data-artifact="sglang.yaml"]').click()
                expect(page.locator("#artifact-code")).to_contain_text("enable_hierarchical_cache: false")
                page.keyboard.press("End")
                expect(page.locator('[data-artifact="experiment.json"]')).to_be_focused()
                plan = json.loads(page.locator("#artifact-code").inner_text())
                assert not plan["validation"]["errors"]
                page.locator("#play").click()
                expect(page.locator("#play")).to_have_attribute("aria-pressed", "true")
                page.locator("#play").click()
                # Language navigation retains the same page and fragment.
                page.evaluate("location.hash='lab'")
                page.locator(".lang-toggle-btn").click()
                other = "zh" if lang == "en" else "en"
                page.wait_for_url(f"{base}/{other}/#lab")
                page.locator("#reset").click()
                for width in (1440, 390, 320):
                    page.set_viewport_size({"width": width, "height": 900})
                    for step in range(6):
                        page.locator(f'[data-step="{step}"]').click()
                        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (lang, width, step)
                        if width <= 720:
                            # The main illustration must be visible in the page, without
                            # discovering or opening the floating preview button first.
                            expect(page.locator(".workbench > .preview #world")).to_be_visible()
                            expect(page.locator("#world .world-lesson")).to_have_attribute("data-lesson", str(step))
                            preview_box = page.locator(".workbench > .preview").bounding_box()
                            builder_box = page.locator(".builder-panel").bounding_box()
                            assert preview_box["y"] + preview_box["height"] <= builder_box["y"]
                            page.locator("#mobile-preview").click()
                            expect(page.locator("#mobile-preview-dialog")).to_be_visible()
                            expect(page.locator("#world")).to_be_visible()
                            assert page.locator("#mobile-preview-dialog").evaluate(
                                "el => el.scrollWidth <= el.clientWidth"
                            ), (width, step)
                            page.locator("#close-preview").click()
                            expect(page.locator("#mobile-preview-dialog")).not_to_be_visible()
                            expect(page.locator(".workbench > .preview #world")).to_be_visible()
                page.locator('[data-step="0"]').click()
                page.locator('[name="task"][value="custom"]').check()
                expect(page.locator('.workbench [data-signal="custom"]')).to_be_visible()
                page.locator("#undo").click()
                expect(page.locator('.workbench [data-signal="math"]')).to_be_visible()
                # External fleets with PD/cache must also fit narrow preview panels.
                page.locator('[data-step="1"]').click()
                page.locator('[name="layout"][value="external"]').check()
                page.locator('[data-step="4"]').click()
                page.locator('[name="pd"]').check()
                page.locator('[name="hicache"]').check()
                page.locator("#mobile-preview").click()
                expect(page.locator(".external-map .sampler")).to_have_count(3)
                assert page.locator("#mobile-preview-dialog").evaluate("el => el.scrollWidth <= el.clientWidth")
                clusters = page.locator(".external-cluster").evaluate_all(
                    "nodes => nodes.map(n => {const r=n.getBoundingClientRect();return {top:r.top,bottom:r.bottom};})"
                )
                assert clusters[0]["bottom"] < clusters[1]["top"] < clusters[1]["bottom"] < clusters[2]["top"]
                page.locator("#close-preview").click()
                expect(page.locator(".workbench .external-map .sampler")).to_have_count(3)
                for guide in ("policy-mismatch", "rl-systems"):
                    page.locator(f'.library-teaser a[href="advanced/{guide}.html"]').click()
                    expect_rendered_math(page)
                    assert page.locator("#learn-dialog").evaluate("el => el.scrollWidth <= el.clientWidth")
                    page.keyboard.press("Escape")
                # Check local page/anchor targets from the new landing page and guides.
                for doc in (
                    "index",
                    "get_started/experiment-guide",
                    "advanced/policy-mismatch",
                    "advanced/rl-systems",
                    "advanced/parallelism-memory",
                    "advanced/rollout-scheduling",
                ):
                    page.goto(f"{base}/{lang}/{doc}.html", wait_until="domcontentloaded")
                    links = page.locator("a[href]").evaluate_all("(nodes) => nodes.map(a => a.href)")
                    for link in links:
                        url = urlsplit(link)
                        if not link.startswith(base) or url.query:
                            continue
                        file = build / unquote(url.path.lstrip("/"))
                        if file.is_dir():
                            file /= "index.html"
                        assert file.is_file(), link
                        if url.fragment and file.suffix == ".html" and not url.fragment.startswith("experiment="):
                            # MyST anchors may be percent encoded for Chinese headings.
                            anchor = unquote(url.fragment)
                            html = file.read_text()
                            assert f'id="{anchor}"' in html or f'name="{anchor}"' in html, link
                page.goto(f"{base}/{lang}/library.html")
                expect(page.locator(".library-categories .toctree-l1")).to_have_count(8)
                expect(page.locator(".library-categories .toctree-l2")).to_have_count(0)
                page.locator('.library-categories a[href="operations/index.html"]').click()
                expect(page.locator('article a[href="../developer_guide/profiling.html"]')).to_be_visible()
                query = "parallelism" if lang == "en" else "并行"
                page.goto(f"{base}/{lang}/search.html?q={query}")
                expect(
                    page.locator('#search-results a[href*="advanced/parallelism-memory.html"]').first
                ).to_be_visible()
                assert not errors, errors
                context.close()
                print(f"{lang}: configuration, history, sharing, downloads, mobile and local links passed")
            # Build exactly what release-docs publishes: English at root, Chinese in zh/.
            with tempfile.TemporaryDirectory() as publish:
                docs = build.parent
                for language, destination in (("en", Path(publish)), ("zh", Path(publish) / "zh")):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "sphinx",
                            "-b",
                            "html",
                            "-D",
                            f"language={language}",
                            "--conf-dir",
                            str(docs),
                            "-W",
                            "--keep-going",
                            str(docs / language),
                            str(destination),
                        ],
                        env={**os.environ, "SLIME_DOC_LAYOUT": "root", "SLIME_DOC_LANG": language},
                        cwd=docs,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    assert result.returncode == 0, result.stdout
                for prefix in ("/", "/slime/"):
                    context = browser.new_context()
                    missing = []

                    def production_route(route, _request, *, prefix=prefix, missing=missing):
                        relative = unquote(urlsplit(route.request.url).path.removeprefix(prefix))
                        file = Path(publish) / relative
                        if file.is_dir():
                            file /= "index.html"
                        if not file.is_file():
                            missing.append(route.request.url)
                            route.fulfill(status=404, body="not found")
                        else:
                            route.fulfill(path=file)

                    context.route("https://thudm.github.io/**", production_route)
                    page = context.new_page()
                    page.goto(f"https://thudm.github.io{prefix}#lab", wait_until="networkidle")
                    expect(page.locator('[name="model"]')).to_have_count(6)
                    page.locator('[data-learn="grpo"]').click()
                    expect_rendered_math(page)
                    page.keyboard.press("Escape")
                    page.locator(".lang-toggle-btn").click()
                    page.wait_for_url(f"https://thudm.github.io{prefix}zh/#lab")
                    page.locator('[data-learn="grpo"]').click()
                    expect_rendered_math(page)
                    page.keyboard.press("Escape")
                    page.locator(".lang-toggle-btn").click()
                    page.wait_for_url(f"https://thudm.github.io{prefix}#lab")
                    page.goto(f"https://thudm.github.io{prefix}developer_guide/profiling.html")
                    page.locator(".lang-toggle-btn").click()
                    page.wait_for_url(f"https://thudm.github.io{prefix}zh/developer_guide/profiling.html")
                    assert not missing, missing
                    context.close()
            print(
                "actual production builds: root and https://thudm.github.io/slime/ assets, reader and language links passed"
            )
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    run()
