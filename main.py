from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Tuple
import pulp

app = FastAPI(title="API de Programação Linear Dinâmica")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"],  allow_headers=["*"],
)

def solve_lp(
    nome: str,
    variaveis: List[str],
    cof_obj: Dict[str, float],
    restricoes: List[Tuple[Dict[str, float], str, float]],
) -> Tuple[Dict[str, float], float]:
    prob = pulp.LpProblem(nome, pulp.LpMaximize)
    x = {v: pulp.LpVariable(v, lowBound=0) for v in variaveis}
    prob += pulp.lpSum(cof_obj[v] * x[v] for v in variaveis), "Z"
    for i,(coef,sense,rhs) in enumerate(restricoes,1):
        expr = pulp.lpSum(coef.get(v,0)*x[v] for v in variaveis)
        prob += (expr <= rhs if sense=="<=" else expr >= rhs), f"R{i}"
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = {v: x[v].value() for v in variaveis}
    z = pulp.value(prob.objective)
    return sol,z

class ConstraintSpec(BaseModel):
    coef: Dict[str,float]
    sense: str
    rhs: float

class ModelSpec(BaseModel):
    variables: List[str]
    coef_obj: Dict[str,float]
    constraints: List[ConstraintSpec]

class SolveResult(BaseModel):
    solution: Dict[str,float]
    Z: float
    passo_a_passo: List[str]

@app.post("/api/solve", response_model=SolveResult)
def solve_dynamic(model: ModelSpec):
    sol,z = solve_lp(
        nome="Dinâmico",
        variaveis=model.variables,
        cof_obj=model.coef_obj,
        restricoes=[(c.coef,c.sense,c.rhs) for c in model.constraints]
    )
    # gera passo a passo
    passos=[]
    binds=[]
    for idx,c in enumerate(model.constraints):
        lhs=sum(c.coef.get(v,0)*sol[v] for v in model.variables)
        binds.append((idx,c,lhs))
        
    idx0,c0,lhs0=min(binds,key=lambda t:abs(t[2]-t[1].rhs))
    def coef_str(coef):
        terms=[f"{coef[v]}·{v}" for v in model.variables if coef.get(v,0)!=0]
        return " + ".join(terms) if terms else "0"
    passos.append(f"🔒 Restrição mais limitante: {coef_str(c0.coef)} {c0.sense} {c0.rhs} (LHS={lhs0:.2f})")
    best_var=max(model.coef_obj,key=lambda v:model.coef_obj[v])
    passos.append(f"💰 Maior lucro unitário: {best_var} (R${model.coef_obj[best_var]})")
    passos.append(f"✏️ Definir {best_var} = {c0.rhs} e demais = 0")
    for idx,c,lhs in binds:
        passos.append(f"🔒 Restrição {idx+1}: {coef_str(c.coef)} {c.sense} {c.rhs} → LHS={lhs:.2f}")
    terms=" + ".join(f"{model.coef_obj[v]}·{sol[v]}" for v in model.variables)
    passos.append(f"Σ Z: {terms} = R${z:.2f}")
    return SolveResult(solution=sol,Z=z,passo_a_passo=passos)
