// @ts-check
import { defineConfig } from "astro/config";

export default defineConfig({
  site: "https://davidmgray.com",
  base: "/neural-cost",
  trailingSlash: "always",
  markdown: {
    shikiConfig: {
      themes: { light: "github-light", dark: "github-dark-dimmed" },
      defaultColor: false,
      wrap: false,
    },
  },
});
