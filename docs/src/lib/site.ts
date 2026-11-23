export const REPO = "https://github.com/davidgraymi/neural-cost";
export const PYPI = "https://pypi.org/project/neural-cost/";

export const SECTIONS = ["Start", "Guides", "Models", "Reference"] as const;

/** Join a site-relative path onto the configured base, e.g. url("docs/") -> "/neural-cost/docs/". */
export function url(path = ""): string {
  const base = import.meta.env.BASE_URL.replace(/\/+$/, "");
  const clean = path.replace(/^\/+/, "");
  return `${base}/${clean}`;
}
