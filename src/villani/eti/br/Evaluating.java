package villani.eti.br;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.TreeMap;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.core.Instances;

public class Evaluating {

	public static void run(String id, LogBuilder log, TreeMap<String, String> entradas) {

		boolean ehd = Boolean.parseBoolean(entradas.get("ehd"));
		boolean lbp = Boolean.parseBoolean(entradas.get("lbp"));
		boolean sift = Boolean.parseBoolean(entradas.get("sift"));
		boolean gabor = Boolean.parseBoolean(entradas.get("gabor"));
		boolean mlknn = Boolean.parseBoolean(entradas.get("mlknn"));
		boolean brknn = Boolean.parseBoolean(entradas.get("brknn"));
		boolean chain = Boolean.parseBoolean(entradas.get("chain"));
		String[] tecnicas = {"Ehd","Lbp", "Sift","Gabor"};
		String[] eixos = {"T","D","A","B"};
		String[] classificadores = {"mulan.classifier.lazy.MLkNN","mulan.classifier.lazy.BRkNN","mulan.classifier.transformation.ClassifierChain"};
		
		for(String tecnica: tecnicas){

			if(tecnica.equals("Ehd") && !ehd) continue;
			if(tecnica.equals("Lbp") && !lbp) continue;
			if(tecnica.equals("Sift") && !sift) continue;
			if(tecnica.equals("Gabor") && !gabor) continue;

			for(String eixo: eixos){

				for(String classificador: classificadores){

					if(classificador.equals("mulan.classifier.lazy.MLkNN") && !mlknn) continue;
					if(classificador.equals("mulan.classifier.lazy.BRkNN") && !brknn) continue;
					if(classificador.equals("mulan.classifier.transformation.ClassifierChain") && !chain) continue;

					String subBase = id + "-" + tecnica + "-Sub" + 0 + "-" + eixo;

					Instances instancias = null;
					try{
						log.write(" - Desserializando instancias a partir da sub-base " + subBase);
						FileInputStream instanciasFIS = new FileInputStream(subBase + ".bsi");
						ObjectInputStream instanciasOIS = new ObjectInputStream(instanciasFIS);
						instancias = (Instances)instanciasOIS.readObject();
						instanciasOIS.close();
						instanciasFIS.close();
					} catch(Exception e){
						log.write(" - Falha ao desserializar instancias: " + e.getMessage());
						System.exit(0);
					}

					LabelsMetaDataImpl rotulos = null;
					try{
						log.write(" - Desserializando respectiva estrutura de rótulos");
						FileInputStream rotulosFIS = new FileInputStream(subBase + ".labels");
						ObjectInputStream rotulosOIS = new ObjectInputStream(rotulosFIS);
						rotulos = (LabelsMetaDataImpl)rotulosOIS.readObject();
						rotulosOIS.close();
						rotulosFIS.close();
					} catch(Exception e){
						log.write(" - Falha ao desserializar rotulos: " + e.getMessage());
						System.exit(0);
					}

					MultiLabelInstances trainingSet = null;
					try{
						log.write(" - Instanciando conjunto de treinamento multirrótulo");
						trainingSet = new MultiLabelInstances(instancias, rotulos);
					} catch(InvalidDataFormatException idfe){
						log.write(" - Erro no formato de dados ao instanciar conjunto multirrótulos: " + idfe.getMessage());
						System.exit(0);
					}

					MultiLabelLearnerBase mlLearner = null;
					try{
						log.write(" - Instanciando classificador " + classificador);
						mlLearner = (MultiLabelLearnerBase)Class.forName(classificador).newInstance();
					} catch(ClassNotFoundException cnfe){
						log.write(" - A classe " + classificador + " não foi encontrada: " + cnfe.getMessage());
						System.exit(0);
					} catch(IllegalAccessException iae){
						log.write(" - A classe " + classificador + " não pode ser acessada: " + iae.getMessage());
						System.exit(0);
					} catch(InstantiationException ie){
						log.write(" - Não foi possível instanciar um objeto da classe " + classificador + ": " + ie.getMessage());
						System.exit(0);
					}

					try{
						log.write(" - Construindo modelo do " + classificador + " a partir do conjunto de treinamento " + subBase);
						mlLearner.build(trainingSet);
					} catch(Exception e){
						log.write(" - Falha ao construir o modelo do classificador: " + e.getMessage());
						System.exit(0);
					}

					log.write(" - Instanciando avaliador");
					Evaluator avaliador = new Evaluator();

					log.write(" - Instanciando lista de medidas");
					ArrayList<Measure> medidas = new ArrayList<Measure>();
					medidas.add(new HammingLoss());
					medidas.add(new SubsetAccuracy());
					medidas.add(new ExampleBasedPrecision());
					medidas.add(new ExampleBasedRecall());
					medidas.add(new ExampleBasedFMeasure());
					medidas.add(new ExampleBasedAccuracy());
					medidas.add(new ExampleBasedSpecificity());
					int numOfLabels = trainingSet.getNumLabels();
					medidas.add(new MicroPrecision(numOfLabels));
					medidas.add(new MicroRecall(numOfLabels));
					medidas.add(new MicroFMeasure(numOfLabels));
					medidas.add(new AveragePrecision());
					medidas.add(new Coverage());
					medidas.add(new OneError());
					medidas.add(new IsError());
					medidas.add(new ErrorSetSize());
					medidas.add(new RankingLoss());

					for(int i = 1; i < 10; i++){

						String subBaseTeste = id + "-" + tecnica + "-Sub" + i + "-" + eixo;

						try{
							log.write(" - Desserializando instancias a partir da sub-base " + subBaseTeste);
							FileInputStream instanciasFIS = new FileInputStream(subBaseTeste + ".bsi");
							ObjectInputStream instanciasOIS = new ObjectInputStream(instanciasFIS);
							instancias = (Instances)instanciasOIS.readObject();
							instanciasOIS.close();
							instanciasFIS.close();
						} catch(Exception e){
							log.write(" - Falha ao desserializar instancias: " + e.getMessage());
							System.exit(0);
						}

						try{
							log.write(" - Desserializando respectiva estrutura de rótulos");
							FileInputStream rotulosFIS = new FileInputStream(subBaseTeste + ".labels");
							ObjectInputStream rotulosOIS = new ObjectInputStream(rotulosFIS);
							rotulos = (LabelsMetaDataImpl)rotulosOIS.readObject();
							rotulosOIS.close();
							rotulosFIS.close();
						} catch(Exception e){
							log.write(" - Falha ao desserializar rotulos: " + e.getMessage());
							System.exit(0);
						}

						MultiLabelInstances testSet = null;
						try{
							log.write(" - Instanciando conjunto de teste multirrótulo");
							testSet = new MultiLabelInstances(instancias, rotulos);
						} catch(InvalidDataFormatException idfe){
							log.write(" - Erro no formato de dados ao instanciar conjunto multirrótulos: " + idfe.getMessage());
							System.exit(0);
						}

						log.write(" - Avaliando o modelo gerado pelo classificador " + classificador);
						Evaluation avaliacao = null;
						try {
							avaliacao = avaliador.evaluate(mlLearner, testSet, medidas);
						} catch (IllegalArgumentException iae){
							log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
							System.exit(0);
						} catch (Exception e) {
							log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
							System.exit(0);
						}

						log.write(" - Salvando resultado da avaliação");
						File resultado = new File(subBaseTeste + ".result");
						try{
							FileWriter escritor = new FileWriter(resultado);
							escritor.write("=> Avaliação do " + classificador + "\n\n");
							escritor.write(avaliacao.toString());
							escritor.close();
						} catch(IOException ioe){
							log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
							System.exit(0);
						}

					}

				}

			}

		}

	}

}
