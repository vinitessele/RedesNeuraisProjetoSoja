import os
import json
import time
from datetime import datetime
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# -------------------- CONFIGURAÇÕES --------------------
# ⚠️ Substitua pela sua chave via variável de ambiente (NÃO no código!)
# Para segurança, use: export GEMINI_API_KEY="sua_chave_aqui" no terminal
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCEEV7xAv0Ugpzvt_sMLOgzh-050zlvV4I")
genai.configure(api_key=API_KEY)

INPUT_DIR = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\separar"
LOG_FILE = "classificacao_log.json"

# 🔧 CORREÇÃO: Use o nome completo do modelo ou alternativa
MODEL_NAME = "gemini-1.5-flash-latest"  # Tente também: "gemini-1.5-pro-latest"
MAX_RETRIES = 3

# -------------------- FUNÇÕES --------------------
def listar_modelos_disponiveis():
    """Lista modelos disponíveis para debug"""
    try:
        print("🔍 Verificando modelos disponíveis...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"   ✓ {m.name}")
        print()
    except Exception as e:
        print(f"⚠️ Erro ao listar modelos: {e}\n")


def get_mime_type(filename):
    """Retorna o MIME type baseado na extensão do arquivo"""
    ext = filename.lower()
    if ext.endswith(".png"):
        return "image/png"
    elif ext.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return None


def classify_image(model, image_path, filename):
    """Classifica a imagem usando o modelo Gemini"""
    mime_type = get_mime_type(filename)
    if not mime_type:
        return None, "Formato não suportado"

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        return None, f"Erro ao ler arquivo: {e}"

    # Prompt otimizado para classificação
    prompt = [
        {
            "mime_type": mime_type,
            "data": image_bytes
        },
        (
            "Você é um agrônomo especialista em fenologia da soja.\n\n"
            "Analise a imagem e identifique o estágio vegetativo DOMINANTE da planta de soja.\n\n"
            "ESTÁGIOS POSSÍVEIS:\n"
            "- VE: Emergência (cotilédones acima do solo)\n"
            "- VC: Cotilédones abertos\n"
            "- V1: Primeiro nó com folhas unifolioladas\n"
            "- V2: Segundo nó com primeira folha trifoliolada\n"
            "- V3: Terceiro nó com segunda folha trifoliolada\n"
            "- VN: Estágio vegetativo avançado (V4 ou superior)\n"
            "- OUTRO: Não é possível identificar ou não é soja\n\n"
            "RESPONDA APENAS COM UM DOS CÓDIGOS ACIMA (VE, VC, V1, V2, V3, VN ou OUTRO).\n"
            "Se houver dúvida entre dois estágios, escolha o mais avançado."
        )
    ]

    config = GenerationConfig(
        temperature=0.1,
        max_output_tokens=20,
        top_p=0.8,
        top_k=10
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(prompt, generation_config=config)

            if not response or not response.text:
                return "OUTRO", "Resposta vazia da API"

            # Processa a resposta
            stage = response.text.strip().upper()
            valid_stages = ["VE", "VC", "V1", "V2", "V3", "VN", "OUTRO"]

            # Busca por estágio válido na resposta
            for v in valid_stages:
                if v in stage:
                    return v, None

            return "OUTRO", f"Resposta não reconhecida: {stage}"

        except Exception as e:
            error_msg = str(e)
            print(f"   ⚠️ Tentativa {attempt + 1}/{MAX_RETRIES} falhou: {error_msg}")
            
            if attempt + 1 == MAX_RETRIES:
                return None, f"Erro após {MAX_RETRIES} tentativas: {error_msg}"
            
            # Backoff exponencial
            time.sleep(2 ** attempt)


def rename_file(image_path, filename, stage):
    """Renomeia o arquivo com o prefixo do estágio"""
    dir_path = os.path.dirname(image_path)
    name, ext = os.path.splitext(filename)
    
    new_name = f"{stage}_{filename}"
    new_path = os.path.join(dir_path, new_name)
    
    # Evita conflitos de nome
    counter = 1
    while os.path.exists(new_path):
        new_name = f"{stage}_{name}_{counter}{ext}"
        new_path = os.path.join(dir_path, new_name)
        counter += 1
    
    os.rename(image_path, new_path)
    return new_name


def save_log(results):
    """Salva o log de processamento em JSON"""
    log = {
        "data_processamento": datetime.now().isoformat(),
        "modelo_usado": MODEL_NAME,
        "total_imagens": len(results),
        "sucesso": sum(1 for r in results if r["status"] == "sucesso"),
        "erros": sum(1 for r in results if r["status"] != "sucesso"),
        "resultados": results,
    }
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Log salvo em: {LOG_FILE}")


# -------------------- PROCESSAMENTO PRINCIPAL --------------------
def main():
    print("🌱 Classificador de Estágios Vegetativos de Soja")
    print("=" * 60)
    print(f"📁 Diretório: {INPUT_DIR}")
    print(f"🤖 Modelo: {MODEL_NAME}\n")

    # Lista modelos disponíveis (útil para debug)
    listar_modelos_disponiveis()

    # Inicializa o modelo
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"✅ Modelo {MODEL_NAME} carregado com sucesso!\n")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("\n💡 Tente um destes modelos:")
        print("   - gemini-1.5-flash-latest")
        print("   - gemini-1.5-pro-latest")
        print("   - gemini-pro-vision")
        return

    # Busca imagens
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Diretório não encontrado: {INPUT_DIR}")
        return

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    
    total = len(image_files)
    if total == 0:
        print(f"⚠️ Nenhuma imagem encontrada em {INPUT_DIR}")
        return

    print(f"📊 Encontradas {total} imagens para classificar.\n")
    print("-" * 60)

    results = []
    start_time = time.time()

    # Processa cada imagem
    for idx, filename in enumerate(image_files, 1):
        print(f"\n[{idx}/{total}] 📸 {filename}")
        image_path = os.path.join(INPUT_DIR, filename)

        stage, error = classify_image(model, image_path, filename)

        if error:
            print(f"   ❌ {error}")
            results.append({
                "arquivo_original": filename,
                "status": "erro",
                "erro": error
            })
            continue

        # Renomeia o arquivo
        try:
            new_name = rename_file(image_path, filename, stage)
            print(f"   ✅ Classificado como: {stage}")
            print(f"   📝 Novo nome: {new_name}")
            
            results.append({
                "arquivo_original": filename,
                "arquivo_novo": new_name,
                "estagio": stage,
                "status": "sucesso"
            })
        except Exception as e:
            print(f"   ⚠️ Erro ao renomear: {e}")
            results.append({
                "arquivo_original": filename,
                "estagio": stage,
                "status": "erro_renomear",
                "erro": str(e)
            })

    # Finalização
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 CLASSIFICAÇÃO CONCLUÍDA!")
    print(f"⏱️  Tempo total: {elapsed:.2f}s")
    print(f"✅ Sucessos: {sum(1 for r in results if r['status'] == 'sucesso')}")
    print(f"❌ Erros: {sum(1 for r in results if r['status'] != 'sucesso')}")
    
    save_log(results)


if __name__ == "__main__":
    main()