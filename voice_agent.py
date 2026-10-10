import sys
from ollama import chat
from agent import MODEL, SYSTEM_PROMPT, TOOLS, FUNCTIONS
from stt import SpeechRecognizer

def main():
    stt = SpeechRecognizer(model_size="base", device="cpu", compute_type="int8")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 55)
    print("🎙️  AI Voice & Desktop Agent Aktif (Linux)")
    print("=" * 55)
    print("- Tekan [Enter] langsung untuk Bicara (Speech-to-Text).")
    print("- Atau ketik teks secara manual jika tidak ingin bicara.")
    print("- Ketik 'exit' untuk keluar.\n")

    while True:
        try:
            prompt_guide = "Kamu [Enter untuk Bicara / Teks] > "
            user_input = input(prompt_guide).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("exit", "quit", "keluar"):
            break

        # Jika pengguna hanya menekan Enter tanpa mengetik teks -> Aktifkan Rekam Suara
        if not user_input:
            try:
                audio = stt.record_until_silence()
                if len(audio) == 0:
                    print("⚠️  Tidak ada suara yang terdeteksi.\n")
                    continue
                print("⏳ Mengubah suara menjadi teks...")
                user_input = stt.transcribe(audio, language="id")
                print(f"🗣️  Terdengar: \"{user_input}\"\n")
            except Exception as e:
                print(f"❌ Error STT: {e}\n")
                continue

            if not user_input:
                print("⚠️  Suara tidak terbaca dengan jelas.\n")
                continue

        messages.append({
            "role": "user",
            "content": user_input,
        })

        # Siklus iterasi pemanggilan agent dan tools
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
                        print(f"⚙️  [Tool] {name}: {arguments}")
                        result = function(**arguments)
                    except Exception as exc:
                        result = f"Tool gagal: {exc}"

                print(f"📋 [Hasil] {result}")

                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                })
        else:
            print("⚠️  Batas putaran tercapai.")

    print("\nAgent dihentikan.")


if __name__ == "__main__":
    main()
