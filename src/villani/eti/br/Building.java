package villani.eti.br;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.HashSet;
import java.util.TreeMap;

import mulan.data.InvalidDataFormatException;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;

public class Building {

	public static void run(String id, LogBuilder log, TreeMap<String, String> entradas){

		boolean ehd = Boolean.parseBoolean(entradas.get("ehd"));
		boolean lbp = Boolean.parseBoolean(entradas.get("lbp"));
		boolean sift = Boolean.parseBoolean(entradas.get("sift"));
		boolean gabor = Boolean.parseBoolean(entradas.get("gabor"));
		String rotulos = entradas.get("rotulos");
		String[] tecnicas = {"Ehd","Lbp", "Sift","Gabor"};
		String[] eixos = {"T","D","A","B"};

		for(String tecnica: tecnicas){
			if(tecnica.equals("Ehd") && !ehd) continue;
			if(tecnica.equals("Lbp") && !lbp) continue;
			if(tecnica.equals("Sift") && !sift) continue;
			if(tecnica.equals("Gabor") && !gabor) continue;
			for(String eixo: eixos){
				for(int i = 0; i < 10; i++){

					String base = tecnica + "/" + tecnica + "-Sub" + i + ".arff";
					String subBase = id + tecnica + "-Sub" + i + "-" + eixo;
					MultiLabelInstances mlData = null;

					try{
						
						log.write(" - Instanciando conjunto multirrótulo a partir da base " + base);
						mlData = new MultiLabelInstances(base, rotulos);
						
					} catch(InvalidDataFormatException idfe){
						
						log.write(" - Erro no formato ao criar conjunto multirrótulo: " + idfe.getMessage());
						System.exit(0);
						
					}

					log.write(" - Criando filtro de rótulos");
					LabelsMetaDataImpl estruturaRotulos = (LabelsMetaDataImpl)mlData.getLabelsMetaData();
					HashSet<String> filtros = new HashSet<String>();
					for(String rotulo: estruturaRotulos.getLabelNames()){
						if(!rotulo.startsWith(eixo)) filtros.add(rotulo);
					}
					
					log.write(" - Removendo das instancias os rótulos que não pertencem ao eixo " + eixo);
					Instances instancias = mlData.getDataSet();
					for(String filtro: filtros){
						instancias.deleteAttributeAt(instancias.attribute(filtro).index());
					}
					
					
					try{
						// Remove os atributos rótulos de outros eixos
						log.write(" - Reajustando a estrutura de rótulos ao novo conjunto de rótulos das instancias");
						mlData = mlData.reintegrateModifiedDataSet(instancias);
						
					} catch(InvalidDataFormatException idfe){
						
						log.write(" - Erro ao reajustar conjunto multirrótulo: " + idfe.getMessage() + " : " + idfe.getCause());
						System.exit(0);
						
					}

					try{
						
						log.write(" - Serializando amostras da técnica " + tecnica + " para o eixo " + eixo);
						FileOutputStream amostrasFOS = new FileOutputStream(subBase + ".bsi");
						ObjectOutputStream amostrasOOS = new ObjectOutputStream(amostrasFOS);
						amostrasOOS.writeObject(mlData.getDataSet());
						amostrasOOS.flush();
						amostrasOOS.close();
						amostrasFOS.flush();
						amostrasFOS.close();
						
					} catch(Exception e){
						
						log.write(" - Falha ao serializar conjunto de dados: " + e.getMessage());
						System.exit(0);
						
					}

					try{
						
						log.write(" - Serializando estrutura de rótulos para o respectivo conjunto");
						FileOutputStream rotulosFOS = new FileOutputStream(subBase + ".labels");
						ObjectOutputStream rotulosOOS = new ObjectOutputStream(rotulosFOS);
						rotulosOOS.writeObject(mlData.getLabelsMetaData());
						rotulosOOS.flush();
						rotulosOOS.close();
						rotulosFOS.flush();
						rotulosFOS.close();
						
					} catch(Exception e){
						
						log.write(" - Falha ao serializar estrutura de rótulos do conjunto multirrótulo: " + e.getMessage());
						System.exit(0);
						
					}

				}
			}
		}
	}
}
