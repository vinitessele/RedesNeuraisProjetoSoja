import os

path = r"D:\DriverGoogle\VNT - Sistemas\ZeraBank\imagens"

print("Existe?", os.path.exists(path))
print("É diretório?", os.path.isdir(path))

if os.path.exists(path):
    print("Conteúdo:", os.listdir(path))
else:
    print("❌ Caminho não encontrado!")
