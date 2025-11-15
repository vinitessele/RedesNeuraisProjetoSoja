import os
import json
import time
import base64
from datetime import datetime
from anthropic import Anthropic
from PIL import Image
import io

# -------------------- CONFIGURAÇÕES --------------------
# ⚠️ Substitua pela sua chave via variável de ambiente (NÃO no código!)
# Para segurança, use: export ANTHROPIC_API_KEY="sua_chave_aqui" no terminal
API_KEY = os.environ.get("")
client = Anthropic(api_key=API_KEY)

INPUT_DIR = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\separar"
LOG_FILE = "classificacao_log.json"

# 🔧 Modelo Claude
MODEL_NAME = "claude-sonnet-4-5-20250929"  # Ou "claude-opus-4-20250514" para maior precisão
MAX_RETRIES = 3
# Base64 aumenta ~33% o tamanho, então usamos 3.7MB como limite para garantir que fique abaixo de 5MB
MAX_FILE_SIZE_MB = 3.7  

# -------------------- FUNÇÕES --------------------
def get_mime_type(filename):
    """Retorna o MIME type baseado na extensão do arquivo"""
    ext = filename.lower()
    if ext.endswith(".png"):
        return "image/png"
    elif ext.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    elif ext.endswith(".webp"):
        return "image/webp"
    elif ext.endswith(".gif"):
        return "image/gif"
    return None


def compress_and_encode_image(image_path, max_file_size_mb=MAX_FILE_SIZE_MB):
    """
    Comprime a imagem se necessário e retorna em base64.
    Considera que base64 aumenta ~33% o tamanho.
    Retorna: (image_data_base64, mime_type, foi_comprimida, tamanho_final_mb)
    """
    max_size_bytes = max_file_size_mb * 1024 * 1024
    original_size = os.path.getsize(image_path)
    
    print(f"   📏 Tamanho do arquivo: {original_size / 1024 / 1024:.2f} MB")
    
    # Se o arquivo original já é pequeno, tenta usar direto
    if original_size <= max_size_bytes:
        try:
            with open(image_path, "rb") as f:
                file_data = f.read()
            
            # Verifica o tamanho em base64
            encoded = base64.standard_b64encode(file_data).decode("utf-8")
            encoded_size = len(encoded)
            encoded_size_mb = encoded_size / 1024 / 1024
            
            print(f"   📊 Tamanho em base64: {encoded_size_mb:.2f} MB")
            
            # Se o base64 ainda está ok, usa a imagem original
            if encoded_size_mb < 4.9:  # Margem de segurança
                mime_type = get_mime_type(image_path)
                return encoded, mime_type, False, encoded_size_mb
            
            print(f"   ⚠️ Base64 muito grande! Comprimindo...")
        except Exception as e:
            print(f"   ⚠️ Erro ao ler arquivo: {e}")
    
    # Precisa comprimir
    print(f"   🗜️ Comprimindo imagem...")
    
    try:
        # Abre a imagem
        img = Image.open(image_path)
        original_format = img.format
        original_size_img = img.size
        
        print(f"   📐 Resolução original: {original_size_img[0]}x{original_size_img[1]}")
        
        # Converte para RGB se necessário
        if img.mode in ('RGBA', 'P', 'LA'):
            print(f"   🎨 Convertendo {img.mode} para RGB...")
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Estratégia 1: Tentar diferentes níveis de qualidade
        quality = 95
        best_result = None
        
        while quality >= 20:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            compressed_data = buffer.read()
            
            # Testa o tamanho em base64
            encoded = base64.standard_b64encode(compressed_data).decode("utf-8")
            encoded_size_mb = len(encoded) / 1024 / 1024
            
            if encoded_size_mb < 4.9:  # Deixa margem de segurança
                print(f"   ✅ Comprimida: qualidade {quality}%, {encoded_size_mb:.2f} MB em base64")
                return encoded, "image/jpeg", True, encoded_size_mb
            
            # Guarda o melhor resultado até agora
            if best_result is None or encoded_size_mb < best_result[2]:
                best_result = (encoded, "image/jpeg", encoded_size_mb)
            
            quality -= 10
        
        # Estratégia 2: Redimensionar se ainda está grande
        print(f"   🔄 Redimensionando imagem...")
        
        # Calcula novo tamanho (reduz em 30% por vez)
        scale_factor = 0.7
        attempts = 0
        max_attempts = 5
        
        while attempts < max_attempts:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            
            if new_width < 400 or new_height < 400:
                print(f"   ⚠️ Imagem muito pequena ({new_width}x{new_height}), usando melhor resultado anterior")
                if best_result:
                    return best_result[0], best_result[1], True, best_result[2]
                break
            
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Tenta salvar com qualidade razoável
            for q in [85, 75, 65, 55]:
                buffer = io.BytesIO()
                resized.save(buffer, format='JPEG', quality=q, optimize=True)
                buffer.seek(0)
                compressed_data = buffer.read()
                
                encoded = base64.standard_b64encode(compressed_data).decode("utf-8")
                encoded_size_mb = len(encoded) / 1024 / 1024
                
                if encoded_size_mb < 4.9:
                    print(f"   ✅ Redimensionada: {new_width}x{new_height}, qualidade {q}%, {encoded_size_mb:.2f} MB")
                    return encoded, "image/jpeg", True, encoded_size_mb
            
            scale_factor -= 0.1
            attempts += 1
        
        # Se chegou aqui, usa o melhor resultado que conseguiu
        if best_result:
            print(f"   ⚠️ Usando melhor compressão possível: {best_result[2]:.2f} MB")
            return best_result[0], best_result[1], True, best_result[2]
        
        return None, None, False, 0
        
    except Exception as e:
        print(f"   ❌ Erro ao comprimir: {e}")
        return None, None, False, 0


def classify_image(image_path, filename):
    """Classifica a imagem usando o modelo Claude"""
    # Comprime e codifica a imagem
    image_data, mime_type, was_compressed, final_size = compress_and_encode_image(image_path)
    
    if image_data is None:
        return None, "Erro ao processar imagem"

    # Prompt otimizado para classificação
    prompt = """Você é um agrônomo especialista em fenologia da soja.

Analise a imagem e identifique o estágio fenológico DOMINANTE da planta de soja.

ESTÁGIOS VEGETATIVOS:
- VE: Emergência (cotilédones acima do solo)
- VC: Cotilédones completamente abertos
- V1: Primeiro nó, folhas unifolioladas desenvolvidas
- V2: Segundo nó, primeira folha trifoliolada desenvolvida
- V3: Terceiro nó, segunda folha trifoliolada desenvolvida
- V4: Quarto nó, terceira folha trifoliolada desenvolvida
- V5: Quinto nó, quarta folha trifoliolada desenvolvida
- V6: Sexto nó, quinta folha trifoliolada desenvolvida
- VN: Estágio vegetativo avançado (V7 ou superior)

ESTÁGIOS REPRODUTIVOS:
- R1: Início do florescimento (uma flor aberta em qualquer nó)
- R2: Florescimento pleno (flor aberta em um dos dois últimos nós)
- R3: Início da formação de vagens (vagem de 5mm em um dos 4 últimos nós)
- R4: Vagem cheia (vagem de 2cm em um dos 4 últimos nós)
- R5: Início do enchimento de grãos (grão de 3mm em vagem dos 4 últimos nós)
- R6: Vagem cheia (grãos verdes preenchendo cavidade da vagem)
- R7: Início da maturação (uma vagem madura na planta)
- R8: Maturação plena (95% das vagens maduras)
- R9: Ponto de colheita (maturação fisiológica completa)

- OUTRO: Não é possível identificar ou não é soja

RESPONDA APENAS COM UM DOS CÓDIGOS ACIMA.
Se houver dúvida entre dois estágios, escolha o mais avançado."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=50,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            if not response or not response.content:
                return "OUTRO", "Resposta vazia da API"

            # Processa a resposta
            stage = response.content[0].text.strip().upper()
            valid_stages = [
                "VE", "VC", "V1", "V2", "V3", "V4", "V5", "V6", "VN",
                "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
                "OUTRO"
            ]

            # Busca por estágio válido na resposta (do mais específico ao mais geral)
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

    return None, "Falha após todas as tentativas"


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
        "imagens_comprimidas": sum(1 for r in results if r.get("foi_comprimida", False)),
        "resultados": results,
    }
    
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Log salvo em: {LOG_FILE}")


# -------------------- PROCESSAMENTO PRINCIPAL --------------------
def main():
    print("🌱 Classificador de Estágios Vegetativos de Soja v2.0")
    print("=" * 60)
    print(f"📁 Diretório: {INPUT_DIR}")
    print(f"🤖 Modelo: {MODEL_NAME}")
    print(f"📏 Limite arquivo: {MAX_FILE_SIZE_MB} MB (base64: ~{MAX_FILE_SIZE_MB * 1.33:.1f} MB)\n")

    # Verifica a chave API
    if not API_KEY or API_KEY == "sua_chave_aqui":
        print("❌ ERRO: Configure a variável ANTHROPIC_API_KEY")
        print("💡 Use: export ANTHROPIC_API_KEY='sua_chave_aqui'")
        return

    print(f"✅ Cliente Claude inicializado!\n")

    # Verifica se o Pillow está instalado
    try:
        from PIL import Image
    except ImportError:
        print("❌ ERRO: Biblioteca Pillow não encontrada")
        print("💡 Instale com: pip install Pillow")
        return

    # Busca imagens
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Diretório não encontrado: {INPUT_DIR}")
        return

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))
        and not f.startswith(("VE_", "VC_", "V1_", "V2_", "V3_", "V4_", "V5_", "V6_", "VN_",
                              "R1_", "R2_", "R3_", "R4_", "R5_", "R6_", "R7_", "R8_", "R9_", "OUTRO_"))
    ]
    
    total = len(image_files)
    if total == 0:
        print(f"⚠️ Nenhuma imagem não processada encontrada em {INPUT_DIR}")
        return

    print(f"📊 Encontradas {total} imagens para classificar.\n")
    print("-" * 60)

    results = []
    start_time = time.time()

    # Processa cada imagem
    for idx, filename in enumerate(image_files, 1):
        print(f"\n[{idx}/{total}] 📸 {filename}")
        image_path = os.path.join(INPUT_DIR, filename)

        stage, error = classify_image(image_path, filename)

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
    if total > 0:
        print(f"⚡ Média: {elapsed/total:.2f}s por imagem")
    print(f"✅ Sucessos: {sum(1 for r in results if r['status'] == 'sucesso')}")
    print(f"❌ Erros: {sum(1 for r in results if r['status'] != 'sucesso')}")
    print(f"🗜️  Imagens comprimidas: {sum(1 for r in results if r.get('foi_comprimida', False))}")
    
    save_log(results)


if __name__ == "__main__":
    main()