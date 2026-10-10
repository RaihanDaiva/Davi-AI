import webbrowser
import subprocess
from urllib.parse import urlparse
from ollama import chat

MODEL = "qwen3:4b"

SYSTEM_PROMPT = """
Kamu adalah asisten AI lokal pintar untuk pengguna komputer Linux.
Jawab selalu dalam bahasa Indonesia yang ramah, sopan, dan jelas.

PENTING TENTANG PENGGUNA:
Pengguna adalah orang awam yang tidak mengetahui perintah terminal (bash/Linux).
Pengguna akan berbicara atau bertanya dalam bahasa percakapan sehari-hari.
Tugasmu adalah memahami maksud pertanyaan pengguna dan secara otomatis memanggil tool yang sesuai untuk mendapatkan informasi riil dari sistem komputer.

Panduan pemetaan pertanyaan pengguna:
- Jika pengguna bertanya tentang siapa dirinya / nama pengguna / akun sistem ("siapakah nama saya?", "siapa saya?", "user komputer ini siapa?"):
  Panggil tool `run_command` dengan argumen {"command": "whoami"}.
- Jika pengguna bertanya tentang lokasi folder / direktori saat ini ("saya ada di folder mana?", "di mana posisi saya sekarang?", "cek lokasi direktori"):
  Panggil tool `run_command` dengan argumen {"command": "pwd"}.
- Jika pengguna bertanya tentang hari / tanggal / waktu ("sekarang hari apa?", "tanggal berapa hari ini?", "jam berapa sekarang?"):
  Panggil tool `run_command` dengan argumen {"command": "date"}.
- Jika pengguna bertanya tentang sistem operasi / kernel / info komputer ("sistem operasi apa ini?", "informasi sistem Linux"):
  Panggil tool `run_command` dengan argumen {"command": "uname"}.
- Jika pengguna ingin membuka YouTube atau memutar video ("buka youtube", "buka yt"):
  Panggil tool `open_youtube`.
- Jika pengguna ingin membuka link web tertentu ("buka google", "buka website x"):
  Panggil tool `open_website`.

Aturan eksekusi:
1. Panggil tool secara langsung tanpa memberi tahu rencana pemanggilan sebagai teks.
2. Tunggu hasil eksekusi dari tool, lalu sampaikan jawaban ramah kepada pengguna berdasarkan output tool tersebut.
3. Jangan mengarang atau menebak informasi sistem sebelum memeriksa melalui tool.
"""

# Hanya perintah yang secara eksplisit diizinkan.
ALLOWED_COMMANDS = {
    "whoami": ["whoami"],
    "pwd": ["pwd"],
    "date": ["date"],
    "uname": ["uname", "-a"],
}


def run_command(command: str) -> str:
    """Jalankan perintah Linux dasar untuk mendapatkan informasi sistem komputer.

    Args:
        command: Perintah yang dipilih berdasarkan maksud pengguna:
            - 'whoami': Untuk mengetahui nama pengguna / akun sistem saat ini (misal: 'siapakah nama saya?', 'siapa saya?').
            - 'pwd': Untuk mengetahui lokasi folder/direktori aktif saat ini ('saya di folder mana?').
            - 'date': Untuk mengetahui hari, tanggal, atau jam saat ini ('hari apa sekarang?', 'tanggal berapa?').
            - 'uname': Untuk mengetahui informasi sistem operasi / kernel Linux.
    """
    command = command.strip().lower()
    if command not in ALLOWED_COMMANDS:
        return (
            "Ditolak: perintah tidak ada dalam daftar izin. "
            "Perintah yang diizinkan: "
            + ", ".join(ALLOWED_COMMANDS)
        )

    try:
        result = subprocess.run(
            ALLOWED_COMMANDS[command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0:
            return f"Perintah gagal: {result.stderr.strip()}"

        return result.stdout.strip() or "(Tidak ada output)"

    except subprocess.TimeoutExpired:
        return "Perintah dihentikan karena melewati batas waktu."


def open_website(url: str) -> str:
    """Buka website HTTP atau HTTPS di browser default Linux.

    Args:
        url: URL website yang ingin dibuka.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return "Ditolak: URL harus menggunakan HTTP atau HTTPS."

    try:
        success = webbrowser.open(url)

        if success:
            return f"Permintaan membuka website dikirim: {url}"

        return "Browser tidak berhasil dibuka."

    except Exception as exc:
        return f"Gagal membuka website: {exc}"


def open_youtube() -> str:
    """Buka halaman utama YouTube di browser default ketika pengguna ingin membuka YouTube atau menonton video."""
    return open_website("https://www.youtube.com")


TOOLS = [run_command, open_website, open_youtube]

FUNCTIONS = {
    "run_command": run_command,
    "open_website": open_website,
    "open_youtube": open_youtube,
}


def main():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("AI Desktop Agent aktif.")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        try:
            user_input = input("Kamu > ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("exit", "quit", "keluar"):
            break

        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": user_input,
        })

        # Batasi jumlah putaran agar agent tidak berjalan tanpa akhir.
        for _ in range(5):
            response = chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                think=True,
                options={
                    "num_predict": 1024,
                    "num_ctx": 1024,
                },
            )

            messages.append({
                "role": "assistant",
                "content": response.message.content or "",
                "tool_calls": response.message.tool_calls,
            })

            tool_calls = response.message.tool_calls or []

            if not tool_calls:
                content = response.message.content or ""
                print(f"\nQwen > {content.strip()}\n")
                break

            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments

                function = FUNCTIONS.get(name)

                if function is None:
                    result = f"Tool tidak dikenal: {name}"
                else:
                    try:
                        print(f"[Tool] {name}: {arguments}")
                        result = function(**arguments)
                    except Exception as exc:
                        result = f"Tool gagal: {exc}"

                print(f"[Hasil] {result}")

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                })
        else:
            print("Agent berhenti: batas putaran tercapai.")

    print("\nAgent dihentikan.")


if __name__ == "__main__":
    main()
