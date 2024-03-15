from IPython import embed
import re

def reformat_sums(string):
    signs = [""] + [f" {sign.strip()} " for sign in re.findall("\s*[+-]\s*", string)]
    terms = re.split(r"\s*[+-]\s*", string)

    new_string = ""
    
    num_terms = len(terms)
    for i, term in enumerate(terms):
        indicies = re.findall(r"\\sum_{([a-z])}", term)
        indicies = "".join(indicies)

        term = (re.sub(r"\\sum_{[a-z]}", "", term)).strip()

        if len(indicies) > 0:
            term = r"\sum_{" + indicies + "} " + term

        new_string += f"{signs.pop(0)}" + term
    
    return new_string

def pretty(latex_str, einst=False):
    # Remove sums and order indicator
    if einst:
        latex_str = re.sub(r"\\sum_{.*?}", "", latex_str)
    else:
        latex_str = latex_str.replace(r" \in O", "")
        latex_str = latex_str.replace(r" \in V", "")
        latex_str = reformat_sums(latex_str)
    
    latex_str = latex_str.replace("^{1}", "")
    latex_str = latex_str.replace("^{2}", "")
    latex_str = latex_str.replace("^1", "")
    latex_str = latex_str.replace("^2", "")

    # Fromat indicies
    latex_str = re.sub(r"t_{([a-h]),([i-o])}", r"t^{\1}_{\2}", latex_str)
    latex_str = re.sub(r"t_{([a-h]),([a-h]),([i-o]),([i-o])}", r"t^{\1\2}_{\3\4}", latex_str)

    latex_str = re.sub(r"\\lambda_{([a-h]),([i-o])}", r"\\lambda^{\2}_{\1}", latex_str)
    latex_str = re.sub(r"\\lambda_{([a-h]),([a-h]),([i-o]),([i-o])}", r"\\lambda^{\3\4}_{\1\2}", latex_str)
    
    latex_str = re.sub(r"u_{([a-z]),([a-z]),([a-z]),([a-z])}", r"u^{\1\2}_{\3\4}", latex_str)
    latex_str = re.sub(r"P", r"\\hat{P}", latex_str)

    return latex_str

    # latex_str = re.sub(r"t\^{\d}_{(.*?),(.*?),(.*?),(.*?)}", r"t^{\1\2}_{\3\4}", latex_str)
    # latex_str = re.sub(r"u_{(.*?),(.*?),(.*?),(.*?)}", r"u^{\1\2}_{\3\4}", latex_str)
    # latex_str = re.sub(r"f_{(.*?),(.*?)}", r"f_{\1\2}", latex_str).strip()
def main():
    latex_str = "P(ij) \\lambda^1_{a,i}  \\lambda^1_{b,j}  - \\sum_{k \\in O} \\sum_{c \\in V} \\lambda^1_{c,k}  t^{1}_{c,k}  \\lambda^2_{a,b,i,j}  - \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V}  \\frac{1}{4} \\lambda^2_{a,b,i,j}  \\lambda^2_{c,d,k,l}  t^{2}_{c,d,k,l}  - \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V}  \\frac{1}{4} \\lambda^2_{a,b,k,l}  \\lambda^2_{c,d,i,j}  t^{2}_{c,d,k,l}   + \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V} \\frac{1}{2} t^{1}_{c,k}  t^{1}_{d,l}  \\lambda^2_{a,b,i,j}  \\lambda^2_{c,d,k,l}   + \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V} \\frac{1}{2} t^{1}_{c,k}  t^{1}_{d,l}  \\lambda^2_{a,b,k,l}  \\lambda^2_{c,d,i,j}  - \\sum_{k \\in O} \\sum_{c \\in V}  P(ab) \\lambda^1_{a,k}  t^{1}_{c,k}  \\lambda^2_{b,c,i,j}  - \\sum_{k \\in O} \\sum_{c \\in V}  P(ij) \\lambda^1_{c,i}  t^{1}_{c,k}  \\lambda^2_{a,b,j,k}  - \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V}  P(ij) \\lambda^2_{a,c,i,k}  \\lambda^2_{b,d,j,l}  t^{2}_{c,d,k,l}  - \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V}  \\frac{P(ij)}{2} \\lambda^2_{a,b,i,l}  \\lambda^2_{c,d,j,k}  t^{2}_{c,d,k,l}  - \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V}  \\frac{P(ab)}{2} \\lambda^2_{a,d,i,j}  \\lambda^2_{b,c,k,l}  t^{2}_{c,d,k,l}   + \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V} P(ij) t^{1}_{c,k}  t^{1}_{d,l}  \\lambda^2_{a,b,i,l}  \\lambda^2_{c,d,j,k}   + \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V} P(ij) t^{1}_{c,k}  t^{1}_{d,l}  \\lambda^2_{a,c,i,k}  \\lambda^2_{b,d,j,l}   + \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V} P(ab) t^{1}_{c,k}  t^{1}_{d,l}  \\lambda^2_{a,d,i,j}  \\lambda^2_{b,c,k,l}  - \\sum_{k \\in O} \\sum_{c \\in V}  P(ab) P(ij) \\lambda^1_{a,i}  t^{1}_{c,k}  \\lambda^2_{b,c,j,k}  - \\sum_{k \\in O} \\sum_{l \\in O} \\sum_{c \\in V} \\sum_{d \\in V}  P(ij) t^{1}_{c,l}  t^{1}_{d,k}  \\lambda^2_{a,c,i,k}  \\lambda^2_{b,d,j,l} "

    pretty(latex_str)
if __name__ == '__main__':
    main()