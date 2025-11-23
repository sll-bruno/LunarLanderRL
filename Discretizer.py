import numpy as np

class Discretizador:
    """
    Classe responsável pela discretização das variáveis contínuas em estados.
    Gerado por IA Generativa
    """
    def __init__(self):
        # 1. Definição das "Cercas" (Bins)
        # O np.digitize usa esses valores como divisórias.
        
        # X: Focamos nos limites da zona de pouso (1.5)
        self.x_bins = np.array([-1.5, 1.5]) 
        # Resultado: 0=Esq, 1=Alvo, 2=Dir
        
        # Y: Focamos na proximidade do chão
        self.y_bins = np.array([0.1, 0.5, 1.0]) 
        # Resultado: 0=Chão, 1=Baixo, 2=Médio, 3=Alto
        
        # Vx: Focamos na estabilidade (0.5 é o limite do crash)
        self.vx_bins = np.array([-0.5, 0.5]) 
        # Resultado: 0=Rápido Esq, 1=Estável, 2=Rápido Dir
        
        # Vy: Focamos na velocidade de queda (-0.5 é o limite do crash)
        # Adicionei -0.1 para diferenciar "caindo devagar" de "quase parado/subindo"
        self.vy_bins = np.array([-0.5, -0.1]) 
        # Resultado: 0=Crash, 1=Descida Segura, 2=Subindo/Pairando

        # 2. Cálculo do número total de estados
        # O número de buckets é len(bins) + 1
        self.n_x = len(self.x_bins) + 1
        self.n_y = len(self.y_bins) + 1
        self.n_vx = len(self.vx_bins) + 1
        self.n_vy = len(self.vy_bins) + 1
        
        self.n_states = self.n_x * self.n_y * self.n_vx * self.n_vy

    def get_state_index(self, continuous_state):
        """
        Recebe: vetor [x, y, vx, vy]
        Retorna: int único (0 a n_states-1)
        """
        x, y, vx, vy = continuous_state
        
        # Descobre em qual bucket cada variável cai
        x_idx = np.digitize(x, self.x_bins)
        y_idx = np.digitize(y, self.y_bins)
        vx_idx = np.digitize(vx, self.vx_bins)
        vy_idx = np.digitize(vy, self.vy_bins)
        
        # Flattening (Achatamento)
        # Transforma coordenadas multidimensionais (i,j,k,l) em um índice linear
        # Formula: idx = i + (j * Ni) + (k * Ni * Nj) + ...
        
        return (x_idx, y_idx, vx_idx, vy_idx)