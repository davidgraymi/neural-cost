# neural-cost website

Astro site published to GitHub Pages at <https://davidmgray.com/neural-cost/> by
`.github/workflows/pages.yml` on pushes to `main` that touch `docs/`. Pull
requests build it without deploying.

```bash
cd docs
npm ci
npm run dev      # http://localhost:4321/neural-cost/
npm run build    # writes docs/dist/
```

## Adding a page

Create a Markdown file in `src/content/docs/`. Its file name becomes the URL
(`src/content/docs/foo.md` is served at `/neural-cost/docs/foo/`) and the
frontmatter places it in the sidebar:

```yaml
---
title: "Page title"
description: "One sentence shown under the title and in search results."
section: "Guides"   # Start | Guides | Models | Reference
order: 7            # position within the section
---
```

Use `##` and `###` headings; they feed the "On this page" table of contents.
Link between pages with relative URLs such as `../attribution/`. Other pages
(not docs) go in `src/pages/` as `.astro` files.
