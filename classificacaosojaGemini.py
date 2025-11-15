import os
import base64 # Mantido da sua estrutura original
import json
from datetime import datetime
# from pathlib import Path # Removido para manter sua estrutura com 'os'
from google import genai
from google.genai.errors import APIError
from google.genai import types # Importa types para usar Part
import time # Importado para o caso de limite de taxa

# --- Configurações Iniciais ---
# ATENÇÃO: Esta chave está publicamente comprometida. 
# Por favor, revogue-a e crie uma nova.
os.environ["GEMINI_API_KEY"] = "" 

# Configurações do diretório
# INPUT_DIR = "/content/drive/MyDrive/VNT - Sistemas/ZeraBank/separar"
INPUT_DIR = r"D:\GoogleDriver\VNT - Sistemas\ZeraBank\separar"
LOG_FILE = "classificacao_log.json"
MAX_RETRIES = 3 

# Inicializa o cliente Gemini
# O cliente procurará a chave na variável de ambiente GEMINI_API_KEY
try:
    client = genai.Client()
except Exception as e:
    print(f"Erro ao inicializar o cliente Gemini. Certifique-se de que a variável de ambiente GEMINI_API_KEY está definida. Erro: {e}")
    exit()

# Modelo a ser usado
MODEL_NAME = "gemini-pro-vision" # <-- AJUSTE 1: Nome do modelo corrigido

# Lista para armazenar resultados
results = []

def get_mime_type(filename):
    """Determina o tipo MIME da imagem"""
    ext = filename.lower()
    if ext.endswith('.png'):
        return "image/png"
    elif ext.endswith(('.jpg', '.jpeg')):
        return "image/jpeg"
    return None

def classify_image(image_path, filename):
    """Classifica uma única imagem usando a API Gemini"""
    
    mime_type = get_mime_type(filename)
    if not mime_type:
        return None, "Formato não suportado"

    try:
        # Lê a imagem
        with open(image_path, "rb") as img_file:
            image_data_bytes = img_file.read()
            
        # Cria o objeto Part para o Gemini com os dados binários
        image_part = types.Part.from_bytes(
            data=image_data_bytes,
            mime_type=mime_type
        )
        
        # PROMPT OTIMIZADO PARA MAIOR PRECISÃO
        system_instruction = (
            "Você é um agrônomo especialista em fenologia da soja. Sua única tarefa é identificar o estágio vegetativo dominante na imagem. "
            "Use apenas os códigos abaixo e nada mais. Se houver dúvida, escolha o estágio mais avançado.\n"
            "Estágios:\n"
            "- VE: Emergência (cotilédones começando a emergir do solo)\n"
            "- VC: Cotilédones abertos e expandidos\n"
            "- V1: Primeira folha trifoliolada completamente desenvolvida\n"
            "- V2: Segunda folha trifoliolada\n"
            "- V3: Terceira folha trifoliolada\n"
            "- VN: Estágios vegetativos mais avançados (V4+)\n"
            "- OUTRO: Não é soja, foto ruim ou não classificável.\n\n"
            "Responda APENAS e EXCLUSIVAMENTE com um dos códigos: VE, VC, V1, V2, V3, VN ou OUTRO."
        )
        
        prompt = [
            "Classifique o estágio fenológico da soja:",
            image_part
        ]

        # Configuração da geração para forçar a resposta curta e consistente
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,  # Baixa temperatura para respostas consistentes
            max_output_tokens=10 # Limita o tamanho da resposta
        )

        # <-- AJUSTE 2: Configuração para desativar filtros de segurança
        safety_settings = {
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }

        # Loop de tentativas (mantendo a estrutura original de tratamento de erro)
        for attempt in range(MAX_RETRIES):
            try:
                # Chamada à API Gemini com o ajuste de segurança
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=config,
                  )
                
                # TRATAMENTO DE ERRO 'NoneType'
                if response.text is None:
                    error_message = "Resposta vazia (mesmo com filtros desativados)."
                    if response.candidates and response.candidates[0].finish_reason.name == "SAFETY":
                        error_message = "Resposta bloqueada por segurança (mesmo com BLOCK_NONE)."
                    return 'OUTRO', error_message
                
                # Prossiga APENAS se houver texto
                stage = response.text.strip().upper()

                # Valida a resposta
                valid_stages = ['VE', 'VC', 'V1', 'V2', 'V3', 'VN', 'OUTRO']
                if stage not in valid_stages:
                    # Tenta extrair um estágio válido da resposta
                    for valid in valid_stages:
                        if valid in stage:
                            stage = valid
                            break
                    else:
                        stage = 'OUTRO' # Se não encontrar, classifica como OUTRO
                
                return stage, None # Sucesso
            
            except APIError as e:
                # Trata erros da API, incluindo limite de taxa (de forma genérica)
                print(f"   ⚠️  Erro de API (tentativa {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt + 1 == MAX_RETRIES:
                    return None, f"Erro na API do Gemini após {MAX_RETRIES} tentativas: {str(e)}"
                time.sleep(2 ** attempt) # Espera 1s, 2s, 4s

    except Exception as e:
        # Erros como falha de leitura do arquivo, etc.
        return None, f"Erro inesperado: {str(e)}"

def rename_file(image_path, filename, stage):
    """Renomeia o arquivo com o estágio classificado"""
    new_name = f"{stage}_{filename}"
    new_path = os.path.join(os.path.dirname(image_path), new_name)
    
    # Evita sobrescrever arquivos
    counter = 1
    while os.path.exists(new_path):
        name, ext = os.path.splitext(filename)
        new_name = f"{stage}_{name}_{counter}{ext}"
        new_path = os.path.join(os.path.dirname(image_path), new_name)
        counter += 1
    
    os.rename(image_path, new_path)
    return new_name

def save_log(results):
    """Salva log das classificações"""
    # Adicionando status 'erro_resposta_vazia' para melhor contagem de erros no log
    total_errors = sum(1 for r in results if r['status'] in ('erro', 'erro_renomear', 'erro_renomear_apos_vazio', 'erro_resposta_vazia'))
    
    log_data = {
        "data": datetime.now().isoformat(),
        "total": len(results),
        "sucesso": sum(1 for r in results if r['status'] == 'sucesso'),
        "erros": total_errors,
        "resultados": results
    }
    
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

# Processamento principal
print("🌱 Iniciando classificação de imagens de soja com Gemini...\n")

# Lista todas as imagens
try:
    image_files = [f for f in os.listdir(INPUT_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
except FileNotFoundError:
    print(f"❌ Erro: O diretório de entrada não foi encontrado: {INPUT_DIR}")
    exit()

total = len(image_files)
print(f"📊 Encontradas {total} imagens para classificar no modelo {MODEL_NAME}\n")

# Processa cada imagem
for idx, filename in enumerate(image_files, 1):
    image_path = os.path.join(INPUT_DIR, filename)
    print(f"[{idx}/{total}] 📸 Processando {filename}...")
    
    # Tenta classificar
    stage, error = classify_image(image_path, filename)
    
    if error:
        # Se for um erro que já retornou o stage ('OUTRO') no erro do NoneType:
        if stage and stage == 'OUTRO':
            print(f"   ❌ Classificação falhou: {error}")
            try:
                new_name = rename_file(image_path, filename, stage)
                print(f"   ✅ Classificado como {stage} → {new_name}")
                results.append({
                    "arquivo_original": filename,
                    "arquivo_novo": new_name,
                    "estagio": stage,
                    "status": "erro_resposta_vazia",
                    "erro": error 
                })
            except Exception as e:
                print(f"   ⚠️  Classificado como {stage}, mas erro ao renomear: {e}")
                results.append({
                    "arquivo_original": filename,
                    "estagio": stage,
                    "status": "erro_renomear_apos_vazio",
                    "erro": f"{error} | Erro Renomear: {str(e)}"
                })
        
        # Se for um erro que retornou None (APIError ou exceção geral):
        else:
             print(f"   ❌ Erro: {error}")
             results.append({
                 "arquivo_original": filename,
                 "status": "erro",
                 "erro": error
             })
        continue
    
    # Renomeia o arquivo (caminho de sucesso)
    try:
        new_name = rename_file(image_path, filename, stage)
        print(f"   ✅ {stage} → {new_name}")
        
        results.append({
            "arquivo_original": filename,
            "arquivo_novo": new_name,
            "estagio": stage,
            "status": "sucesso"
        })
    except Exception as e:
        print(f"   ⚠️  Classificado como {stage}, mas erro ao renomear: {e}")
        results.append({
            "arquivo_original": filename,
            "estagio": stage,
            "status": "erro_renomear",
            "erro": str(e)
        })

# Salva log
save_log(results)

# Resumo final
print("\n" + "="*50)
print("🎉 CLASSIFICAÇÃO CONCLUÍDA!")
print("="*50)

# Estatísticas por estágio
stages_count = {}
total_erros = sum(1 for r in results if r['status'] in ('erro', 'erro_renomear', 'erro_renomear_apos_vazio'))
total_sucesso_ou_vazio = sum(1 for r in results if r['status'] in ('sucesso', 'erro_resposta_vazia'))

for r in results:
    if r['status'] in ('sucesso', 'erro_resposta_vazia'):
        stage = r['estagio'] if r['estagio'] else 'OUTRO'
        stages_count[stage] = stages_count.get(stage, 0) + 1

print("\n📈 Distribuição por estágio (inclui 'OUTRO' de erros de resposta):")
for stage, count in sorted(stages_count.items()):
    print(f"   {stage}: {count} imagens")

print(f"\n✅ Classificados com sucesso: {sum(1 for r in results if r['status'] == 'sucesso')}")
print(f"⚠️  OUTRO (Respostas vazias/segurança): {sum(1 for r in results if r['status'] == 'erro_resposta_vazia')}")
print(f"❌ Erros graves (API/Renomeio): {total_erros}")
