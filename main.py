from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pulp

app = FastAPI(title="API de Programação Linear")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetalhesEntrega(BaseModel):
    questao: str
    variaveis_decisao: dict
    dados: dict
    funcao_objetivo: str
    restricoes: list[str]
    passo_a_passo: list[str]
    metodo: str
    resultado: dict
    historia: Optional[str] = None

def solve_lp(
    nome: str,
    variaveis: dict[str,str],
    cof_obj: dict[str,float],
    restricoes: list[tuple[dict[str,float], str, float]],
) -> tuple[dict[str,float], float]:
    prob = pulp.LpProblem(nome, pulp.LpMaximize)
    x = {v: pulp.LpVariable(v, lowBound=0) for v in variaveis}
    prob += pulp.lpSum(cof_obj[v] * x[v] for v in variaveis), "Z"
    for i, (coef, sentido, rhs) in enumerate(restricoes, 1):
        expr = pulp.lpSum(coef.get(v, 0) * x[v] for v in variaveis)
        if sentido == "<=":
            prob += expr <= rhs, f"R{i}"
        else:
            prob += expr >= rhs, f"R{i}"
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = {v: x[v].value() for v in variaveis}
    z = pulp.value(prob.objective)
    return sol, z

@app.get("/api/questao1", response_model=DetalhesEntrega)
def questao1():
    questao = "Operação de granel em São Chico: milho (x1) e soja (x2)."
    variaveis = {"x1":"toneladas de milho","x2":"toneladas de soja"}
    cof_obj = {"x1":200,"x2":300}
    restricoes = [
      ({"x1":0.4,"x2":0.5},"<=",120),  # esteira
      ({"x1":0.2,"x2":0.3},"<=",80),   # grua
      ({"x1":1,  "x2":1  },"<=",150),  # pátio
    ]
    passo = [
      "Definir x1,x2 ≥ 0",
      "z = 200·x1 + 300·x2",
      "Impor restrições (esteira, grua, pátio)",
      "Resolver Simplex (CBC)",
      "Ler x* e Z*"
    ]
    historia = (
        "Carla, gerente do porto “São Chico”, recebe navios de milho e soja. "
        "Com 120 h de esteira, 80 h de grua e 150 t de pátio, o Simplex indica "
        "destinar toda a capacidade à soja (150 t), gerando R$45 000/dia."
    )
    sol, z = solve_lp("Q3_Granel", variaveis, cof_obj, restricoes)
    return {
      "questao": questao,
      "variaveis_decisao": variaveis,
      "dados": {"coef_obj": cof_obj},
      "funcao_objetivo": "Max Z = 200·x1 + 300·x2",
      "restricoes": [
        "0.4 x1 + 0.5 x2 ≤ 120",
        "0.2 x1 + 0.3 x2 ≤ 80",
        "1 x1 + 1 x2 ≤ 150",
      ],
      "passo_a_passo": passo,
      "metodo": "Simplex via CBC (PuLP)",
      "resultado": {"x": sol, "Z": z},
      "historia": historia
    }

@app.get("/api/questao2", response_model=DetalhesEntrega)
def questao2():
    questao = "Distribuição multimodal em Joinville: x1…x4."
    variaveis = {
      "x1":"eletrônicos","x2":"móveis",
      "x3":"perecíveis","x4":"construção"
    }
    cof_obj = {"x1":500,"x2":800,"x3":600,"x4":700}
    restricoes = [
      ({"x1":2,  "x2":3,  "x3":1,  "x4":4  },"<=",100),
      ({"x1":0.5,"x2":1,  "x3":0.5,"x4":0.5},"<=",50),
      ({"x1":1,  "x2":2,  "x3":1.5,"x4":2  },"<=",60),
    ]
    passo = [
      "Definir x1…x4 ≥ 0",
      "z = 500·x1 + 800·x2 + 600·x3 + 700·x4",
      "Impor restrições (carga, empilhadeira, motorista)",
      "Resolver Simplex (CBC)",
      "Ler x* e Z*"
    ]
    historia = (
        "Pedro, em Joinville, avalia 4 tipos de remessa. "
        "Com caminhões limitados, o Simplex recomenda 25 viagens de construção "
        "(x4), maximizando R$17 500/dia."
    )
    sol, z = solve_lp("Q2_Joinville", variaveis, cof_obj, restricoes)
    return {
      "questao": questao,
      "variaveis_decisao": variaveis,
      "dados": {"coef_obj": cof_obj},
      "funcao_objetivo": "Max Z = 500·x1 + 800·x2 + 600·x3 + 700·x4",
      "restricoes": [
        "2 x1 + 3 x2 + 1 x3 + 4 x4 ≤ 100",
        "0.5 x1 + 1 x2 + 0.5 x3 + 0.5 x4 ≤ 50",
        "1 x1 + 2 x2 + 1.5 x3 + 2 x4 ≤ 60",
      ],
      "passo_a_passo": passo,
      "metodo": "Simplex via CBC (PuLP)",
      "resultado": {"x": sol, "Z": z},
      "historia": historia
    }

@app.get("/api/questao3", response_model=DetalhesEntrega)
def questao3():
    questao = (
        "Maximização entrega bebidas – Bebidas Maruim envia pallets "
        "de 600 garrafas (700 mL) a 16 bairros; CL já incorporado."
    )
    bairros = [
      "Atiradores","América","Bucarein","Anita Garibaldi",
      "Costa e Silva","Santo Antônio","Bom Retiro","Zona Ind. Norte",
      "Iririú","Comasa","Aventureiro","Itaum",
      "Adhemar Garcia","Boehmerwald","Paranaguamirim","Nova Brasília"
    ]
    variaveis = { f"x{i+1}": bairros[i] for i in range(16) }
    cof_obj = {
      "x1":480,  "x2":475,  "x3":450,  "x4":445,
      "x5":400,  "x6":420,  "x7":395,  "x8":260,
      "x9":380,  "x10":360, "x11":375, "x12":355,
      "x13":380, "x14":340, "x15":295, "x16":320
    }
    dist = {
      "x1":2,  "x2":2,  "x3":2,  "x4":3,
      "x5":5,  "x6":4,  "x7":5,  "x8":12,
      "x9":6,  "x10":7, "x11":6, "x12":6,
      "x13":6, "x14":8, "x15":10,"x16":9
    }
    restricoes = []
    restricoes.append(({v:1 for v in variaveis}, "<=", 800))
    restricoes.append((dist, "<=", 4500))
    regioes = {
      "Central": bairros[0:4],
      "Norte":   bairros[4:8],
      "Leste":   bairros[8:12],
      "Sul":     bairros[12:16]
    }
    min_reg = {"Central":200,"Norte":200,"Leste":130,"Sul":120}
    for reg, membros in regioes.items():
        restricoes.append((
            { f"x{bairros.index(b)+1}": 1 for b in membros },
            ">=", min_reg[reg]
        ))
    passo = [
      "Definir x₁…x₁₆ ≥ 0",
      "z = ∑ CLᵢ·xᵢ",
      "Impor restrições: pallets≤800, km-pallet≤4500",
      "Impor mín. por região",
      "Resolver Simplex (CBC)",
      "Ler x* e Z*"
    ]
    historia = (
      "A Bebidas Maruim envia semanalmente pallets de 600 garrafas "
      "para 16 bairros de Joinville. O lucro líquido por pallet já incorpora "
      "o custo R$20/km. O modelo garante cobertura mínima e maximiza o lucro."
    )
    sol, z = solve_lp("Q3_Bebidas", variaveis, cof_obj, restricoes)
    return {
      "questao": questao,
      "variaveis_decisao": variaveis,
      "dados": {"coef_obj": cof_obj, "distancias": dist},
      "funcao_objetivo": "Max Z = ∑ CLᵢ·xᵢ",
      "restricoes": [
        "∑ xᵢ ≤ 800",
        "∑ distânciaᵢ·xᵢ ≤ 4500",
        *[f"∑ x ∈ {r} ≥ {min_reg[r]}" for r in regioes]
      ],
      "passo_a_passo": passo,
      "metodo": "Simplex via CBC (PuLP)",
      "resultado": {"x": sol, "Z": z},
      "historia": historia
    }
