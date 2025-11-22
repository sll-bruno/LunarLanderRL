import numpy as np  


class LunarLanderEnv:
    """
        Classe do ambiente de RL\n
        ----- Descrição do MDP -----\n
        Estados:
            x: Posição Horizontal
            y: Posição Vertical
            vx: Velocidade vertical 
            vy: Velocidade horizontal
            OBS: Considera-se que os valores dos estados estão centrados em zero, isto é,
            velocidades negativas indicam movimento para esquerda ou para baixo 
        Estados Terminais: (Valores de recompensa podem e possivelmente serão alterados, de inicio ta bem arbitrário)
            Pouso suave:
                |x| <= 1.5 and y = 0 and |vx| < 0.5 and |vy| < 0.5 and t < T_MAX 
            Crash:
                y = 0 and t < T_MAX and (|x| > 1.5 or |vx| > 0.5 or |vy| > 0.5)
            Fim:
                t >= T_MAX or |x| > 4.5 ==> 
        Reward:
            Pouso suave: R = +100
            Crash: R = -1000
            Fim: R = -100
            Para cada time step: -1
            Força no motor principal: -1
            Força nos motores laterais: - 0.5
                
        Ações:
            0: Nenhuma força é aplicada
            1: Força no motor esquerda: Aplica força para direita
            2: Força no motor direito, aplica força para esquerda
            3: Motor principal, aplica força para cima

    """

    def __init__(self, stochastic=False):
        """
        Definição das variáveis e constantes do ambiente
        """
        self.isStochastic = stochastic 
        self.gravity = 1.62  # Gravidade lunar em m/s²
        self.mass = 1000  # Massa do módulo lunar em kg
        self.main_thrust = 2000  # Força do motor principal em N
        self.side_thrust = 10  # Força dos motores laterais em N
        self.dt = 0.04 # segundos

        # Variáveis de estado inicial
        self.x = 0
        self.y = 1.5
        self.vx = 0
        self.vy = -0.5

        self.actions_to_rewards = {
            0: -1,
            1: - 1,
            2: -1.5,
            3: -1
        }

        self.state = None # Estado inicial do ambiente é nulo, mas será definido no reset
        self.reset()

    def reset(self):
        """Reseta o ambiente para o estado inicial.
            Return:
                Tupla: (initialx = 0, initialy = 1.5, initialvx = 0, initialvy = -0.5)
        """

        self.x = 0
        self.y = 1.5
        self.vx = 0
        self.vy = -0.5

        self.state = np.array([self.x, self.y, self.vx, self.vy])
        return self.state

    def step(self, action):
        """
            Contém a dinâmica do ambiente: No nosso caso, o modelo físico que modela o movimento do agente
        """
        # Dado a ação realizada, definimos a força agindo sobre o agente
        # 0: Nenhuma força é aplicada
        # 1: Força no motor esquerda: Aplica força para direita
        # 2: Força no motor direito, aplica força para esquerda
        # 3: Motor principal, aplica força para cima

        match action:
            case 0: 
                force = (0,0)
            case 1:
                force = (+self.side_thrust, 0)
            case 2:
                force = (-self.side_thrust,0)
            case 3:
                force = (0, self.main_thrust, self)

        force_x, force_y = force

        if self.isStochastic: ##
            # Do something
            print("Stochastic modeeee")

        # Obtém acelerações
        acc_x = force_x/self.mass
        acc_y = force_y/self.mass - self.gravity

        # Atualiza velocidades
        self.vx = self.vx + acc_x*self.dt
        self.vy = self.vy + acc_y*self.dt

        # Atualiza a posição
        self.x = self.x + self.vx * self.dt
        self.y = self.y + self.vy * self.dt

        done =  False
        r = self.actions_to_rewards[action] # Já contem penalização por time_step

        # Verificação de estados terminais

        """
            Pouso suave: R = +100
            Crash: R = -1000
            Fim: R = -100
            Para cada time step: -1
            Força no motor principal: -1
            Força nos motores laterais: - 0.5
        """
        
        if self.y <= 0.0 and (abs(self.x) <= 1.5) and (abs(self.vx) <= 0.5) and (abs(self.vy) <= 0.5): 
            # Pouso suave
            done = True
            r += 100
        elif self.y <= 0.0 and (abs(self.x) > 1.5) or (abs(self.vx) > 0.5) or (abs(self.vy) > 0.5):
            # Crash
            done = True
            r -= 1000
        elif abs(self.x) >= 4.5:
            done = True
            r -= 100
            # Out of Limits

        s = get_state()

        return self.state, r, done, info

    def get_state(self):
        return np.array([self.x, self.y, self.vx, self.vy])

    def render(self):
        # Placeholder for rendering the environment
        print(f"Current state: {self.state}")