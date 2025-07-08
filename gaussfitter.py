import torch
from typing import Optional, Sequence

def moments(data: torch.Tensor, circle: int, rotate: int, vheight: int):
    total = data.sum()
    device = data.device
    height, width = data.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    x = (xx * data).sum() / total
    y = (yy * data).sum() / total
    col = data[:, int(y)]
    width_x = torch.sqrt(torch.abs(((torch.arange(col.size, device=device) - y) ** 2 * col).sum() / col.sum()))
    row = data[int(x), :]
    width_y = torch.sqrt(torch.abs(((torch.arange(row.size, device=device) - x) ** 2 * row).sum() / row.sum()))
    width = (width_x + width_y) / 2.0
    height_val = torch.median(data)
    amplitude = data.max() - height_val
    mylist = [amplitude.item(), x.item(), y.item()]
    if vheight == 1:
        mylist = [height_val.item()] + mylist
    if circle == 0:
        mylist = mylist + [width_x.item(), width_y.item()]
    else:
        mylist = mylist + [width.item()]
    if rotate == 1:
        mylist = mylist + [0.0]
    return tuple(mylist)

def twodgaussian(params, circle, rotate, vheight):
    # params: (height, amplitude, x, y, width_x, width_y, rota)
    params = list(params)
    if vheight == 1:
        height = params.pop(0)
    else:
        height = 0.0
    amplitude, center_x, center_y = params.pop(0), params.pop(0), params.pop(0)
    if circle == 1:
        width = params.pop(0)
        width_x = width_y = width
    else:
        width_x, width_y = params.pop(0), params.pop(0)
    if rotate == 1:
        rota = params.pop(0)
        rota = torch.pi / 180.0 * rota
    else:
        rota = 0.0
    def rotgauss(xx, yy):
        if rotate == 1:
            xp = xx * torch.cos(rota) - yy * torch.sin(rota)
            yp = xx * torch.sin(rota) + yy * torch.cos(rota)
        else:
            xp = xx
            yp = yy
        g = height + amplitude * torch.exp(-(((center_x - xp) / width_x) ** 2 + ((center_y - yp) / width_y) ** 2) / 2.0)
        return g
    return rotgauss

def gaussfit(data: torch.Tensor, params: Optional[Sequence[float]] = None, circle: int = 0, rotate: int = 1, vheight: int = 1, max_iter: int = 200):
    device = data.device
    height, width = data.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    if params is None:
        params = moments(data, circle, rotate, vheight)
    params = torch.tensor(params, dtype=torch.float32, device=device, requires_grad=True)
    def model(params):
        return twodgaussian(params, circle, rotate, vheight)(xx, yy)
    optimizer = torch.optim.Adam([params], lr=0.05)
    for _ in range(max_iter):
        optimizer.zero_grad()
        fit = model(params)
        loss = torch.mean((fit - data) ** 2)
        loss.backward()
        optimizer.step()
    return params.detach().cpu().numpy()
