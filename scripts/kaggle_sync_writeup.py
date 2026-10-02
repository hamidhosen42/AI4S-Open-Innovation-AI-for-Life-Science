"""Push writeup/kaggle_writeup.md into the Kaggle Writeup DRAFT through the user's own logged-in Chrome
(macOS AppleScript; needs Chrome > View > Developer > Allow JavaScript from Apple Events).
It clicks Edit, replaces the Project Description, clicks "Save Draft" and verifies. It NEVER submits.

    python scripts/kaggle_sync_writeup.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

DRAFT = "kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/writeups/new-writeup-1789758520935"


def osa(script):
    return subprocess.run(["osascript", "-e", script], capture_output=True, text=True)


def all_tabs():
    r = osa(f'''tell application "Google Chrome"
  set out to ""
  repeat with w from 1 to (count windows)
    repeat with t from 1 to (count tabs of window w)
      if URL of tab t of window w contains "{DRAFT}" then set out to out & (w as text) & "," & (t as text) & ";"
    end repeat
  end repeat
  return out
end tell''')
    return [x for x in r.stdout.strip().split(";") if x]


READY = 'String([...document.querySelectorAll("button")].some(b => ["Edit", "Save Draft"].includes(b.innerText.trim())))'


def find_tab():
    """A loaded draft tab with the Edit / Save Draft button; reload or open one if needed."""
    tabs = all_tabs()
    for loc in reversed(tabs):
        w, t = loc.split(",")
        try:
            if js(w, t, READY) == "true":
                return loc
        except RuntimeError:
            pass
    if tabs:
        w, t = tabs[-1].split(",")
        js(w, t, "location.reload(); 'reloading'")
    else:
        osa(f'tell application "Google Chrome" to make new tab at end of tabs of window 1 with properties {{URL:"https://www.{DRAFT}"}}')
    for _ in range(20):
        time.sleep(2)
        for loc in all_tabs():
            w, t = loc.split(",")
            try:
                if js(w, t, READY) == "true":
                    return loc
            except RuntimeError:
                pass
    sys.exit("no loaded Kaggle draft tab (are you logged in to Kaggle in Chrome?)")


def js(w, t, code):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(code)
    r = subprocess.run(["osascript", "-e", f'set js to read POSIX file "{f.name}" as «class utf8»',
                        "-e", f"tell application \"Google Chrome\" to execute tab {t} of window {w} javascript js"], capture_output=True, text=True)
    os.unlink(f.name)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    return r.stdout.strip()


def click(w, t, label):
    return js(w, t, f'(() => {{ const b = [...document.querySelectorAll("button")].find(x => x.innerText.trim() === {json.dumps(label)}); '
                    f'if (!b) return "missing"; if (b.disabled) return "disabled"; b.click(); return "ok"; }})()')


def main():
    body = open("writeup/kaggle_writeup.md", encoding="utf-8").read()
    w, t = find_tab().split(",")
    has_editor = "String([...document.querySelectorAll('textarea')].some(t => t.getAttribute('aria-label') === 'Project Description' || t.placeholder === 'Project Description'))"
    if js(w, t, has_editor) != "true":
        if click(w, t, "Edit") != "ok":
            sys.exit("could not open the editor (not logged in, or the page changed)")
        for _ in range(15):
            time.sleep(1)
            if js(w, t, has_editor) == "true":
                break
        else:
            sys.exit("the editor did not open")
    res = js(w, t, """(() => {
  const ta = [...document.querySelectorAll('textarea')].find(t => t.getAttribute('aria-label') === 'Project Description' || t.placeholder === 'Project Description');
  if (!ta) return 'no textarea';
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(ta, %s);
  ta.dispatchEvent(new Event('input', {bubbles: true})); ta.dispatchEvent(new Event('change', {bubbles: true}));
  return String(ta.value.length);
})()""" % json.dumps(body))
    if not res.isdigit():
        sys.exit(f"could not fill the description: {res}")
    if click(w, t, "Save Draft") != "ok":
        sys.exit("Save Draft button not available")
    time.sleep(6)
    first_line = next(l for l in body.splitlines() if l.startswith("# ")).lstrip("# ").strip()[:40]
    ok = js(w, t, f"document.body.innerText.includes({json.dumps(first_line)}) && !document.querySelector('textarea[aria-label=\"Project Description\"]')")
    print("Kaggle draft saved (not submitted):", "verified" if ok == "true" else "please check the page", f"| {len(body)} chars")


if __name__ == "__main__":
    main()
