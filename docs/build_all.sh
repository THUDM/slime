#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

echo "[slime-docs] Building EN..."
./build.sh en
echo "[slime-docs] Building ZH..."
./build.sh zh

# Create a lightweight root index with auto redirect based on localStorage (done client side)
ROOT_INDEX=build/index.html
cat > "$ROOT_INDEX" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>slime docs</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body{font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:40px;max-width:720px;margin:auto;color:#222}
    a{color:#0969da;text-decoration:none}a:hover{text-decoration:underline}
    .lang-links{margin-top:1.2rem;display:flex;gap:1rem}
    .note{margin-top:2rem;font-size:12px;color:#666}
  </style>
  <script>
    (function(){
      var stored = null;
      try{stored = localStorage.getItem('slime-doc-lang');}catch(e){}
      var path = (stored === 'zh') ? 'zh/' : (stored === 'en') ? 'en/' : null;
      if(path){ window.location.replace(path); }
    })();
  </script>
</head>
<body>
  <h1>slime Documentation</h1>
  <p>Select language:</p>
  <p class="lang-links"><a href="en/">English</a> <a href="zh/">中文</a></p>
  <p class="note">Auto-redirect uses your last choice if stored; else pick above.</p>
</body>
</html>
EOF

echo "[slime-docs] Done. Root landing page at build/index.html"