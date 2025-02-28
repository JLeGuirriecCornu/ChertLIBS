#! /usr/bin/python3
# -*- coding: utf-8 -*-

########################################################################################
### here some descrition 



########################################################################################

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from math import cos, sin, radians
from matplotlib.backends.backend_pdf import PdfPages


# LIBSS class module 
from LIBS_spectra_class import *


# set the focal measure (yes, globally... :-S ) 
focal_bol = False
focal_measure = 392


class Control_card: 
    def __init__(self):
        self.colormap = {'OK': 'green', 'CAUTION': 'orange', 'WARNING': 'red'}
    # quality function 
    # using parameters (instead of the atributes) for external use proposes and because scalability issues
    def Quality_function(self, Area, mean, std, Previous, verbose = True): 
        # if outseide 3 std
        if Area < (mean - 3*std) or Area > (mean + 3*std): 
            if verbose: print("WARNING: Area outseide 3 std")
            return 'WARNING'
        # if Area and previous[-1] outside 2 std 
        elif len(Previous) > 0 and Area < (mean - 2*std) and Previous[-1] < (mean - 2*std): 
            if verbose: print("WARNING: Area and previous[-1] outside 2 std (down)")
            return 'WARNING'
        elif len(Previous) > 0 and Area > (mean + 2*std) and Previous[-1] > (mean + 2*std): 
            if verbose: print("WARNING: Area and previous[-1] outside 2 std (up)")    
            return 'WARNING'
        # complex (... OR ...) AND (... OR ...) AND (... OR ...)
        elif (len(Previous) > 1 and (Area < (mean - 2*std) or Area > (mean + 2*std)) and 
                                   (Previous[-1] < (mean - 2*std) or Previous[-1] > (mean + 2*std)) and 
                                   (Previous[-2] < (mean - 2*std) or Previous[-2] > (mean + 2*std))): 
            if verbose: print("WARNING: complex (... OR ...) AND (... OR ...) AND (... OR ...)")
            return 'WARNING'
        # if Area and Previous[-1:9] all below or above the mean
        elif len(Previous) > 8 and Area < mean and all(i < mean for i in Previous[-9:]): 
            if verbose: print("WARNING: Area and Previous[-1:9] all below or above the mean (down)")
            return 'WARNING'
        elif len(Previous) > 8 and Area > mean and all(i > mean for i in Previous[-9:]): 
            if verbose: print("WARNING: Area and Previous[-1:9] all below or above the mean (up)")
            return 'WARNING'
        # if Area outside 2 std
        elif Area < (mean - 2*std) or Area > (mean + 2*std): 
            if verbose: print("CAUTION: Area outside 2 std")            
            return 'CAUTION'
        # if Area and Previous[-1:7] all below or above the mean
        elif len(Previous) > 6 and Area < mean and all(i < mean for i in Previous[-7:]): 
            if verbose: print("CAUTION: Area and Previous[-1:7] all below or above the mean (down)")                        
            return 'CAUTION'
        elif len(Previous) > 6 and Area > mean and all(i > mean for i in Previous[-7:]): 
            if verbose: print("CAUTION: Area and Previous[-1:7] all below or above the mean (up)")                                    
            return 'CAUTION'
        # else 
        else: 
            return 'OK'
    # control cord routine. Dicionary Output:
        # keys: shots
        # values: array of assesments, values for each mesaure in shot (key)
    def Control_Card(self, Data, Reference): 
        self.assesments = {}
        # til colum names (shot number) 
        for i in Data.simp_mat.columns:
            ret = []                                            # store the shots  
            # til measures
            for j in Data.simp_mat.index:
                mean = Reference.ref_mean.loc[i]                # mean of reference in shot i 
                std = Reference.ref_std.loc[i]                  # sd of reference in shot i
                data = Data.simp_mat.loc[j,i]                   # inspected shot 
                Previous = list(Data.simp_mat.loc[:(j-1),i])    # previous measures (in shot i)
                # apply Quality_function
                ret.append(self.Quality_function(data, mean, std, Previous))
            # store 
            self.assesments[i] = ret
    # mean control card. Dicionary Output: 
        # keys: measure number
        # value: (mean)assesment of the measure #
    def Mean_Control_Card(self, Data, Reference): 
        # store here 
        self.assesments_mean = {}
        # compute means 
        self.data_mean = Data.simp_mat.transpose(copy=True).mean()
        # til measures
        for j in self.data_mean.index:
            data = self.data_mean.loc[j]                        # inspected measure 
            Previous = list(self.data_mean.loc[:(j-1)])         # previous measures (in shot i)
            self.assesments_mean[j] = self.Quality_function(data, Reference.ref_mean_means, Reference.ref_mean_stds, Previous)
    # plot the mean control card
    def Plot_Control_Card(self, Data, num): 
        y_values = np.array(Data.simp_mat.loc[:,num])
        x_values = np.array(Data.simp_mat.index)
        colors = [self.colormap[cat] for cat in self.assesments[num]] # colors according to the quality function 
        plt.scatter(x_values, y_values, label='Datos', color=colors)
        plt.axhline(y=Reference.ref_mean.loc[num], color='blue', linestyle='-', linewidth=0.8)
        plt.axhline(y=Reference.ref_mean.loc[num] - 2*Reference.ref_std.loc[num], color='red', linestyle='--', linewidth=0.8)
        plt.axhline(y=Reference.ref_mean.loc[num] + 2*Reference.ref_std.loc[num], color='red', linestyle='--', linewidth=0.8)
        plt.axhline(y=Reference.ref_mean.loc[num] - 3*Reference.ref_std.loc[num], color='red', linestyle='--', linewidth=0.8)
        plt.axhline(y=Reference.ref_mean.loc[num] + 3*Reference.ref_std.loc[num], color='red', linestyle='--', linewidth=0.8)
        plt.xlabel(f"Measure in shot {num}")
        #plt.show()
    # plot the dots 
    def Plot_locus(self, Data, num):
        # Obtener el índice según el número dado
        index = list(Data.simp_mat.index).index(num)
        values = [self.assesments[i][index] for i in range(9, 49)]
        colors = [self.colormap[cat] for cat in values]  # Colores según la función de calidad
    
        # Definir el tamaño de la figura
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 4)  # Ancho de la matriz (4 columnas)
        ax.set_ylim(-0.5, 3)  # Alto de la matriz (3 filas)
    
        # Definir el mapa de posiciones personalizado
        custom_positions = [
            (3, 2), (2, 2), (1, 2), (0, 2),  # Primera fila (3, 2) a (0, 2)
            (0, 1), (1, 1), (2, 1), (3, 1),  # Segunda fila (0, 1) a (3, 1)
            (3, 0), (2, 0), (1, 0), (0, 0)   # Tercera fila (3, 0) a (0, 0)
        ]
    
        # Contador de números
        acum = 9
    
        # Dibujar los puntos
        for point_index, (x, y) in enumerate(custom_positions):
            if point_index < 2:
                # Primeros dos loci en gris
                circle = patches.Circle((x, y), 0.4, color="gray")
                ax.add_patch(circle)
            else:
                # Colores del locus focal (cuatro colores por punto)
                start_idx = (point_index - 2) * 4
                point_colors = colors[start_idx:start_idx + 4]
            
                # Construir sectores del círculo y agregar el número
                for k, color in enumerate(point_colors):
                    # Crear sector
                    wedge = patches.Wedge((x, y), 0.4, 90 * k, 90 * (k + 1), color=color)
                    ax.add_patch(wedge)
                
                    # Posición de la etiqueta
                    angle = 90 * k + 45
                    x_text = x + 0.25 * cos(radians(angle))
                    y_text = y + 0.25 * sin(radians(angle))
                    ax.text(x_text, y_text, str(acum), ha='center', va='center', fontsize=8, color="black")
                
                    # Incrementar el número
                    acum += 1
    
        # Construir el gráfico
        ax.axis("off")
        #plt.show()

    def Plot_Mean_Control_Card(self, Data, Reference, num): 
        # Tomar los datos de medición de Data
        measure = np.array(Data.simp_mat.loc[num, :])
        # Datos para el scatter plot
        y_values = np.array(self.data_mean)
        x_values = np.array(self.data_mean.index)
        colors = [self.colormap[cat] for cat in self.assesments_mean.values()]
        # Identificar el índice del nombre `num` en `x_values`
        if num not in x_values:
            raise ValueError(f"La fila con nombre '{num}' no existe en los datos.")
        index_num = np.where(x_values == num)[0][0]  # Encontrar la posición de `num`
        highlighted_x = x_values[index_num]
        highlighted_y = y_values[index_num]
        highlighted_color = colors[index_num]  # Usar el color correspondiente
        # Crear la figura y la rejilla de subplots
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5])
        # Scatter plot principal
        ax_main = plt.subplot(gs[1])
        ax_main.scatter(x_values, y_values, label='Datos', color=colors)
        ax_main.scatter(highlighted_x, highlighted_y, color=highlighted_color, s=100, label='Punto destacado')  # Punto destacado más grande
        # Línea discontinua horizontal desde el eje Y hasta el punto
        ax_main.hlines(highlighted_y, xmin=Data.simp_mat.index[0]-1, xmax=highlighted_x, color='gray', linestyle='--', linewidth=0.8)
        # Líneas horizontales de referencia
        ax_main.axhline(y=Reference.ref_mean_means, color='blue', linestyle='-', linewidth=0.8)
        ax_main.axhline(y=Reference.ref_mean_means - 2 * Reference.ref_mean_stds, color='red', linestyle='--', linewidth=0.8)
        ax_main.axhline(y=Reference.ref_mean_means + 2 * Reference.ref_mean_stds, color='red', linestyle='--', linewidth=0.8)
        ax_main.axhline(y=Reference.ref_mean_means - 3 * Reference.ref_mean_stds, color='red', linestyle='--', linewidth=0.8)
        ax_main.axhline(y=Reference.ref_mean_means + 3 * Reference.ref_mean_stds, color='red', linestyle='--', linewidth=0.8)
        ax_main.set_xlabel(f"Measure {num}")
        # ajustar los valores de X
        ax_main.set_xlim(Data.simp_mat.index[0]-1, ax_main.get_xlim()[1])
        # Ajustar los límites del eje Y
        ymin, ymax = ax_main.get_ylim()
        # Histograma
        ax_hist = plt.subplot(gs[0])
        ax_hist.hist(measure, bins=10, orientation='horizontal', color='gray', edgecolor='black')
        ax_hist.set_ylim(ymin, ymax)    # Establecer los límites del eje Y basados en el scatter plot
        ax_hist.set_xticks([])          # Quitar las etiquetas en X para el histograma
        # Organizar el diseño para evitar superposición
        plt.tight_layout()
        # Mostrar el gráfico o guardarlo según sea necesario
        # plt.show()

def main():
    # reference
    Reference = LIBS_dataset()
    Reference.loadData('./reference/')
    Reference.integrate(Min=False, Max=False)
    Reference.reference_data()
    # data 
    Data = LIBS_dataset()
    Data.loadData('./data/')
    Data.integrate(Min=False, Max=False)
    # take the last of the specified measure 
    if (focal_bol): 
        measure = focal_measure
    else: 
        measure = list(Data.simp_mat.index)[-1]
    # control card 
    Card = Control_card()
    Card.Control_Card(Data, Reference)
    Card.Mean_Control_Card(Data, Reference)
    # plots 
    with PdfPages(f"{measure}_graphs.pdf") as pdf:
        Card.Plot_Mean_Control_Card(Data, Reference, num=measure)
        pdf.savefig()  # Guarda el primer gráfico en el PDF
        plt.close()    # Cierra la figura actual para evitar sobrecargar la memoria
        Card.Plot_locus(Data, num=measure)
        pdf.savefig()  # Guarda el segundo gráfico en el PDF
        plt.close()    # Cierra la segunda figura

if __name__ == "__main__":
    main()

