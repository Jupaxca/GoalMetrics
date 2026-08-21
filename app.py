    # 1. Buscar estrictamente los partidos exactos (Condición y Nivel)
    historial_exacto = df_ordenado[(df_ordenado['Equipo'] == equipo_seleccionado) & 
                                   (df_ordenado['Condición'] == condicion_seleccionada) & 
                                   (df_ordenado['Nivel Rival'] == nivel_seleccionado)].copy()
    
    historial = historial_exacto.copy()
    fuente_datos = f"Exacto ({condicion_seleccionada} vs Nivel {nivel_seleccionado})"
    
    # 2. CRUCE TÁCTICO INTELIGENTE (Si hay 1 exacto, se queda con él y busca solo 1 del contrario)
    if len(historial) < 2:
        condicion_contraria = "Visitante" if condicion_seleccionada == "Local" else "Local"
        
        # Buscamos en la condición contraria el más reciente
        historial_contrario = df_ordenado[(df_ordenado['Equipo'] == equipo_seleccionado) & 
                                          (df_ordenado['Condición'] == condicion_contraria) & 
                                          (df_ordenado['Nivel Rival'] == nivel_seleccionado)].copy()
        
        if len(historial_contrario) == 0:
            historial_contrario = df_ordenado[(df_ordenado['Equipo'] == equipo_seleccionado) & 
                                              (df_ordenado['Condición'] == condicion_contraria)].copy()
        
        if len(historial_contrario) >= 1:
            # Tomamos estrictamente EL ÚLTIMO (1 solo partido) de la otra condición
            ultimo_contrario = historial_contrario.head(1).copy()
            
            # Aplicamos el BAREMO TÁCTICO a ese único partido complementario
            factor_baremo = 0.88 if condicion_seleccionada == "Visitante" else 1.12
            for col in ['Goles', 'Goles Rival', 'Tiros', 'A Puerta', 'Corners', 'Faltas']:
                if col in ultimo_contrario.columns:
                    ultimo_contrario[col] = ultimo_contrario[col] * factor_baremo
            
            # UNIMOS lo que ya teníamos (el exacto que sí existía) + el único complementario ajustado
            historial = pd.concat([historial_exacto, ultimo_contrario])
            fuente_datos = f"Cruce Táctico ({len(historial_exacto)} Exacto + 1 {condicion_contraria} con Baremo)"
        else:
            historial = df_ordenado[df_ordenado['Equipo'] == equipo_seleccionado].head(3).copy()
            fuente_datos = "Emergencia (Historial global mixto por escasez absoluta)"
