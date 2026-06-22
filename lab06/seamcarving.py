# pip install seam-carving
# fonte: https://pypi.org/project/seam-carving/
#
# Seam Carving for Content-Aware Image Resizing (Avidan & Shamir, 2007)
#
# Funcionalidades:
#   - Reduzir a imagem em x e/ou y (remocao de seams)
#   - (a) AUMENTAR a imagem em x e/ou y (insercao de seams)
#   - (b) REMOVER ou PRESERVAR objetos usando mascaras
#
# Exemplos de uso na linha de comando:
#   Reduzir largura para 70% e altura para 90%:
#       python seamcarving.py -i imagens/beach.jpg -x 70 -y 90
#   Aumentar largura para 130%:
#       python seamcarving.py -i imagens/beach.jpg -x 130
#   Remover um objeto (mascara branca sobre o objeto):
#       python seamcarving.py -i imagens/ibiza.jpg --drop_mask mascara_barco.png
#   Reduzir preservando um objeto importante:
#       python seamcarving.py -i imagens/puppy.jpg -x 60 --keep_mask mascara_cao.png

import numpy as np
import cv2
import argparse
import seam_carving


# ---------------------------------------------------------------------------
# Funcoes utilitarias
# ---------------------------------------------------------------------------
def waitKey(window_name, key):
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(1000) & 0xFF
        if keyCode == key:
            cv2.destroyAllWindows()
            break


def carregar_mascara(caminho, shape):
    """Le uma mascara em tons de cinza e a binariza (>127 -> True).

    'shape' e' o (altura, largura) da imagem para garantir o mesmo tamanho.
    """
    mask = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Falha ao ler a mascara: {caminho}")
    if mask.shape[:2] != shape[:2]:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 127


# ---------------------------------------------------------------------------
# (a) Redimensionar (reduzir OU aumentar) em x e y
# ---------------------------------------------------------------------------
def redimensionar(img, x_scale=1.0, y_scale=1.0, energy_mode='backward',
                  order='width-first', keep_mask=None, step_ratio=0.5):
    """Redimensiona a imagem por seam carving.

    x_scale / y_scale: fatores de escala (1.0 = original, <1 reduz, >1 aumenta).
    A funcao seam_carving.resize cobre os dois casos: quando o tamanho alvo e'
    menor ela REMOVE seams; quando e' maior ela INSERE seams (aumento).
    keep_mask: regiao (booleana) a ser preservada durante o processo.
    step_ratio: para aumentos grandes (>50%) o crescimento e' feito em etapas.
    """
    h, w = img.shape[:2]
    new_size = (int(round(w * x_scale)), int(round(h * y_scale)))  # (largura, altura)
    return seam_carving.resize(
        img, new_size,
        energy_mode=energy_mode,
        order=order,
        keep_mask=keep_mask,
        step_ratio=step_ratio)


# ---------------------------------------------------------------------------
# (b) Remover objeto usando mascara
# ---------------------------------------------------------------------------
def remover_objeto(img, drop_mask, keep_mask=None):
    """Remove o objeto coberto por drop_mask, opcionalmente preservando keep_mask.

    Seams sao removidos ate' que toda a regiao marcada desapareca; a imagem
    resultante fica menor na direcao escolhida automaticamente pela biblioteca.
    """
    return seam_carving.remove_object(img, drop_mask=drop_mask, keep_mask=keep_mask)


# ---------------------------------------------------------------------------
# Programa principal (linha de comando)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Seam carving: redimensionamento e remocao/preservacao de objetos")
    ap.add_argument("-i", "--image", required=True,
                    help="Arquivo de imagem a ser processada")
    ap.add_argument("-fe", "--forward", action="store_true",
                    help="Usar energia 'forward' (energia adicionada)")
    ap.add_argument("-hf", "--height_first", action="store_true",
                    help="Remover/inserir primeiro as linhas (height-first)")
    ap.add_argument("-x", "--x_scale", type=int, default=100,
                    help="Percentual de redimensionamento em x (ex.: 70 reduz, 130 aumenta)")
    ap.add_argument("-y", "--y_scale", type=int, default=100,
                    help="Percentual de redimensionamento em y (ex.: 70 reduz, 130 aumenta)")
    ap.add_argument("--keep_mask", default=None,
                    help="Mascara (imagem) da regiao a PRESERVAR")
    ap.add_argument("--drop_mask", default=None,
                    help="Mascara (imagem) da regiao a REMOVER (object removal)")
    ap.add_argument("--step_ratio", type=float, default=0.5,
                    help="Fracao maxima de aumento por etapa (default 0.5)")
    args = vars(ap.parse_args())

    # Imagem
    img = cv2.imread(args["image"])
    if img is None:
        raise ValueError(f"Falha ao ler a imagem: {args['image']}")

    # Modo de energia e ordem
    e_mode = "forward" if args["forward"] else "backward"
    carving_order = "height-first" if args["height_first"] else "width-first"

    # Mascaras (opcionais)
    keep_mask = carregar_mascara(args["keep_mask"], img.shape) if args["keep_mask"] else None
    drop_mask = carregar_mascara(args["drop_mask"], img.shape) if args["drop_mask"] else None

    # (b) Remocao de objeto tem prioridade quando uma drop_mask e' fornecida
    if drop_mask is not None:
        resultado = remover_objeto(img, drop_mask, keep_mask=keep_mask)
    else:
        # (a) Redimensionamento (reduz ou aumenta) em x e y
        resultado = redimensionar(
            img,
            x_scale=args["x_scale"] / 100,
            y_scale=args["y_scale"] / 100,
            energy_mode=e_mode,
            order=carving_order,
            keep_mask=keep_mask,
            step_ratio=args["step_ratio"])

    # Exibicao
    cv2.imshow("Original", img)
    cv2.imshow("Seam Carving", resultado)
    waitKey("Original", 27)
    waitKey("Seam Carving", 27)


if __name__ == "__main__":
    main()
