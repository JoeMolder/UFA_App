/** Converts a team slug like "empire" or "royal-guards" to "Empire" / "Royal Guards" */
export function teamLabel(id: string): string {
  return id.split(/[-_\s]+/).map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
}
