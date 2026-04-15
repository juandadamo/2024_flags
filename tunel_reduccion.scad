// Pieza Lz (ancho = Ly = 470)
Lx = 350;
Ly = 465;
Lreduccion = 100;

a0 = -2 * Lreduccion / pow(Lx, 3);
a1 = 3 * Lreduccion / pow(Lx, 2);

linear_extrude(height = Ly)
    polygon(concat(
        [[0, 0]],                      // inicio piso
        [for (x = [0:5:Lx]) [x, 0]],   // línea del piso
        [for (x = [Lx:-5:0]) [x, a0 * pow(x, 3) + a1 * pow(x, 2)]], // curva superior
        [[0, 0]]                       // cierre
    ));