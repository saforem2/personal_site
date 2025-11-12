import os
import re
import html.parser
from bs4 import BeautifulSoup
from css_html_js_minify import process_single_html_file, html_minify, css_minify, js_minify

# --- Configuration ---
# List of critical CSS files to inline. These should contain styles essential for above-the-fold content.
# Adjust these paths to match your project's CSS files.
CRITICAL_CSS_FILES = [
    "css/profile.css", # Example critical CSS, adjust as needed
    # Add other critical CSS files here
]

# List of CSS files to defer loading. These should be non-critical styles.
# Adjust these paths to match your project's CSS files.
DEFER_CSS_PATTERNS = [
    r"site_libs/bootstrap/.*\.css",
    r"site_libs/quarto-contrib/.*\.css",
    r"static/fonts/.*\.css",
    r"https://fonts.googleapis.com/css.*",
    r"https://cdn.jsdelivr.net/npm/katex@.*\.css",
    r"css/svgbob.css"
]

# List of JavaScript files to defer loading.
# Adjust these patterns to match your project's JS files.
DEFER_JS_PATTERNS = [
    r"site_libs/quarto-contrib/iconify-2.1.0/iconify-icon.min.js",
    r"cdn.jsdelivr.net/npm/jquery@.*\.js",
    r"quarto-preview.js",
    r"site_libs/bootstrap/bootstrap.min.js",
    r"site_libs/quarto-html/anchor.min.js",
    r"site_libs/quarto-html/tippy.umd.min.js",
    r"site_libs/quarto-html/popper.min.js",
    r"site_libs/quarto-html/quarto.js",
    r"site_libs/quarto-listing/list.min.js",
    r"site_libs/quarto-html/axe/axe-check.js",
    r"site_libs/quarto-listing/quarto-listing.js",
    r"site_libs/quarto-html/tabsets/tabsets.js",
    r"site_libs/quarto-search/autocomplete.umd.js",
    r"site_libs/quarto-search/quarto-search.js",
    r"site_libs/quarto-search/fuse.min.js",
    r"site_libs/clipboard/clipboard.min.js",
    r"site_libs/quarto-nav/headroom.min.js",
    r"cdn.jsdelivr.net/npm/katex@.*\.js",
    r"cdn.jsdelivr.net/npm/requirejs@.*\.js",
    r"site_libs/quarto-nav/quarto-nav.js"
]

# List of font file patterns to preload.
FONT_PRELOAD_PATTERNS = [
    r"static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-SemiBoldItalic.woff2",
    r"static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-SemiBold.woff2",
    r"static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-Italic.woff2",
    r"static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-Medium.woff2",
    r"static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-SemiCondensedBold.woff2",
    r"static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-Regular.woff2",
    r"static/fonts/MIosevkaterm/WOFF2/MIosevkaTerm-Bold.woff2",
    r"static/fonts/MIosevkaterm/WOFF2/MIosevkaTerm-Regular.woff2",
    # Add other critical font files here
]

# --- Helper Functions ---

def is_local_file(url, base_dir="."):
    """Checks if a URL points to a local file within the project."""
    if url.startswith("http://") or url.startswith("https://"):
        return False
    # Remove leading slashes for path comparison
    path = url.lstrip('/')
    return os.path.exists(os.path.join(base_dir, path))

def get_file_content(filepath):
    """Reads content from a local file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found for inlining/deferring: {filepath}")
        return None

def apply_optimizations(html_file_path):
    """Applies LCP optimizations to a single HTML file."""
    print(f"Optimizing: {html_file_path}")
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Minify HTML, CSS, JS
    # The css_html_js_minify library handles this directly on files,
    # but for inlining/deferring, we'll manually minify content.
    # For HTML minify, we will apply it at the end.

    # Collect all CSS and JS to be inlined or deferred
    all_link_tags = soup.find_all('link', rel='stylesheet')
    all_script_tags = soup.find_all('script')

    styles_to_inline = []
    links_to_defer = []
    scripts_to_defer = []
    fonts_to_preload = []

    # Process CSS
    for link in all_link_tags:
        href = link.get('href')
        if not href:
            continue

        local_path = href.lstrip('/') # Adjust path for local file system

        # Check for critical CSS to inline
        if href in CRITICAL_CSS_FILES and is_local_file(href):
            css_content = get_file_content(local_path)
            if css_content:
                styles_to_inline.append(css_minify(css_content))
            link.extract() # Remove the link tag
            continue

        # Check for CSS to defer
        should_defer = False
        for pattern in DEFER_CSS_PATTERNS:
            if re.search(pattern, href):
                should_defer = True
                break
        if should_defer:
            links_to_defer.append(link)
            link.extract() # Remove original link
            continue

    # Process JS
    for script in all_script_tags:
        src = script.get('src')
        if not src: # Skip inline scripts for now
            continue

        # Check for JS to defer
        should_defer_js = False
        for pattern in DEFER_JS_PATTERNS:
            if re.search(pattern, src):
                should_defer_js = True
                break
        if should_defer_js:
            scripts_to_defer.append(script)
            script.extract() # Remove original script
            continue

    # Process Font Preload (add as link tags to head)
    head = soup.find('head')
    if head:
        for pattern in FONT_PRELOAD_PATTERNS:
            # Look for existing font links to avoid duplicates
            if not any(re.search(pattern, l.get('href', '')) for l in head.find_all('link', rel='preload')):
                preload_href = None
                # Attempt to find the actual font file by matching the pattern against all possible URLs
                # This part might need refinement based on how your fonts are referenced
                for font_url_in_trace in [
                    "/static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-SemiBoldItalic.woff2",
                    "/static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-SemiBold.woff2",
                    "/static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-Italic.woff2",
                    "/static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-Medium.woff2",
                    "/static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-SemiCondensedBold.woff2",
                    "/static/fonts/MIosevkaQp/WOFF2/MIosevkaQp-Regular.woff2",
                    "/static/fonts/MIosevkaterm/WOFF2/MIosevkaTerm-Bold.woff2",
                    "/static/fonts/MIosevkaterm/WOFF2/MIosevkaTerm-Regular.woff2",
                ]: # Manually list common font URLs or improve pattern matching
                    if re.search(pattern, font_url_in_trace):
                        preload_href = font_url_in_trace
                        break
                if preload_href:
                    new_link = soup.new_tag("link")
                    new_link.attrs = {
                        "rel": "preload",
                        "as": "font",
                        "crossorigin": "anonymous",
                        "href": preload_href
                    }
                    head.append(new_link)

    # Add inlined critical CSS
    if styles_to_inline and head:
        style_tag = soup.new_tag("style")
        style_tag.string = "\n".join(styles_to_inline)
        head.insert(0, style_tag) # Insert at the beginning of head

    # Add deferred CSS links
    if links_to_defer and head:
        for link in links_to_defer:
            original_rel = link.get('rel', ['stylesheet'])[0]
            link['rel'] = 'preload'
            link['as'] = 'style'
            link['onload'] = f"this.onload=null;this.rel='{original_rel}'"
            link.insert_after(soup.new_tag('noscript')).append(link.copy()) # Add noscript fallback
            head.append(link)

    # Add deferred JS scripts (with defer attribute)
    if scripts_to_defer:
        for script in scripts_to_defer:
            script['defer'] = ''
            soup.body.append(script) # Append to body to not block HTML parsing

    # Ensure font-display: swap for all @font-face rules
    # This requires reading and modifying CSS files directly or inline styles.
    # For simplicity, we'll suggest adding it if missing, and assume critical CSS is handled.
    # For external CSS, this would ideally be done at the CSS source.
    # For inline styles, we could attempt to parse and modify.
    # This part is more complex to automate reliably across all scenarios.
    # You might want to consider a dedicated CSS processing tool or manual application.

    # General Minification (if not handled by specific inlining/deferring)
    # Applying html_minify at the end to cover the whole document
    minified_html = html_minify(str(soup))

    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(minified_html)

# --- Post-render hook entry point ---
def quarto_post_render(html_files):
    for f in html_files:
        apply_optimizations(f)

# This block allows the script to be run directly for testing or as a Quarto hook
if __name__ == "__main__":
    # Example usage for testing a single file (replace with your file path)
    # quarto_post_render(["index.html"])
    print("This script is designed to be run as a Quarto post-render hook.")
    print("Refer to the instructions above to integrate it into your project.")
    print("\nSimulating Quarto post-render for all .html files in the current directory:")
    html_files_in_dir = [f for f in os.listdir('.') if f.endswith('.html')]
    if html_files_in_dir:
        quarto_post_render(html_files_in_dir)
    else:
        print("No HTML files found in the current directory for simulation.")

