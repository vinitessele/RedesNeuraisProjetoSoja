import os
from openai import OpenAI, RateLimitError, APIError
import base64
import time

# ⚠️ Mantendo sua chave
os.environ["OPENAI_API_KEY"] = ""

client = OpenAI()
input_dir = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\imagens\R1"

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    if filename.startswith(("VE_", "VC_", "V", "R", "OUTRO_")):
        continue  # evita reprocessar imagens já classificadas

    image_path = os.path.join(input_dir, filename)
    print(f"📸 Classificando {filename}...")

    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    mime_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"

    for attempt in range(10):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um agrônomo especialista em fenologia da soja. "
                            "Classifique a planta da imagem conforme o estágio: "
                            "VE, VC, V1, V2, V3, ... VN, R1, R2, R3, R4, R5, R6, R7 ou R8. "
                            "Retorne somente a fenologia sugerida"
                            "Se não for possível identificar, responda apenas com OUTRO."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Classifique o estágio da soja nesta imagem:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                            },
                        ],
                    },
                ],
                max_tokens=5,  # reduzido para economizar ainda mais
            )
            break

        except RateLimitError:
            wait = 2 + attempt
            print(f"⚠️ Limite atingido — aguardando {wait}s...")
            time.sleep(wait)
        except APIError as e:
            print(f"⚠️ Erro da API: {e}. Tentando novamente em 3s...")
            time.sleep(3)
    else:
        print(f"❌ Falha ao classificar {filename} após várias tentativas.")
        continue

    stage = response.choices[0].message.content.strip().upper().replace(".", "")
    new_name = f"{stage}_{filename}"
    os.rename(image_path, os.path.join(input_dir, new_name))
    print(f"✅ {filename} → {new_name}")

print("\n🎉 Classificação concluída com sucesso!")
