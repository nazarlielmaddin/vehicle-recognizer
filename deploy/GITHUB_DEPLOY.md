# GitHub üzərindən canlı link — 2 yol (Azərbaycanca)

## Yol 1 — Codespaces (1 klik, GitHub-un özündə işləyir) ✅ tövsiyə olunur

1. Repo səhifəsini aç: https://github.com/nazarlielmaddin/vehicle-recognizer
2. Yaşıl **Code** düyməsi → **Codespaces** tabı → **Create codespace on main**
3. 3–5 dəqiqə gözlə (ilk dəfə kitabxanalar + YOLO/CLIP çəkiləri enir, avtomatik)
4. Server avtomatik qalxır. **PORTS** panelində `8000` portunun üzərinə gəl →
   🌐 **Open in Browser** (ya da 🔒 işarəsini **Public** et ki, link hər kəsə açılsın)
5. Link belə olur: `https://<codespace-adı>-8000.app.github.dev`
   — localdakı kimi şəkil yüklə, nəticəni gör.

## Yol 2 — Daimi pulsuz link (Hugging Face Spaces)

1. https://huggingface.co/spaces ünvanında **Create Space** → **Docker** → **Blank**
2. Bu repodakı `deploy/huggingface/Dockerfile` və bütün layihəni ora push et
3. Space avtomatik build olub daimi link verir: `https://huggingface.co/spaces/<adın>/<proyekt>`
4. Kod dəyişəndə: `git push` bəsdir, Space özü yenilənir.
