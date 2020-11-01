using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using Microsoft.ML.Trainers.LightGbm;

namespace WindowsFormsApp3
{
   
    public class ConsumeModel
    {
        public static string PATH_MODEL;
        private static Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictionEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(CreatePredictionEngine);


        public static ModelOutput Predict(ModelInput input)
        {
            ModelOutput result = PredictionEngine.Value.Predict(input);
            return result;
        }

        public static PredictionEngine<ModelInput, ModelOutput> CreatePredictionEngine()
        {          
            MLContext mlContext = new MLContext();
            string modelPath = PATH_MODEL;
            ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            return predEngine;
        }
    }
    public class ModelOutput
    {
        public float Score { get; set; }
    }
    public class ModelInput
    {
        [ColumnName("pic"), LoadColumn(0)]
        public string Pic { get; set; }


        [ColumnName("evaluation"), LoadColumn(1)]
        public float Evaluation { get; set; }


        [ColumnName("expert_square"), LoadColumn(2)]
        public float Expert_square { get; set; }


        [ColumnName("background_expert"), LoadColumn(3)]
        public float Background_expert { get; set; }


        [ColumnName("sample_square"), LoadColumn(4)]
        public float Sample_square { get; set; }


        [ColumnName("background_sample"), LoadColumn(5)]
        public float Background_sample { get; set; }


        [ColumnName("excessive_square"), LoadColumn(6)]
        public float Excessive_square { get; set; }


        [ColumnName("missing_square"), LoadColumn(7)]
        public float Missing_square { get; set; }


        [ColumnName("joint_square"), LoadColumn(8)]
        public float Joint_square { get; set; }
    }

    public static class ModelBuilder
    {

        private static MLContext mlContext = new MLContext(seed: 1);

        public static void CreateModel(string TRAIN_DATA_FILEPATH)
        {
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TRAIN_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: ';',
                                            allowQuoting: true,
                                            allowSparse: false);

            IEstimator<ITransformer> trainingPipeline = BuildTrainingPipeline(mlContext);

            ITransformer mlModel = TrainModel(mlContext, trainingDataView, trainingPipeline);

            Evaluate(mlContext, trainingDataView, trainingPipeline);

            SaveModel(mlContext, mlModel, Path.Combine(Environment.CurrentDirectory, "models",$@"MLModel__{DateTime.Now.ToString($"HHmmss_ddMMyy")}.zip"), trainingDataView.Schema);
        }

        public static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
        {

            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "expert_square", "background_expert", "sample_square", "background_sample", "excessive_square", "missing_square", "joint_square" });
            var trainer = mlContext.Regression.Trainers.LightGbm(labelColumnName: "evaluation", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            return trainingPipeline;
        }

        public static ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Training  model ===============");

            ITransformer model = trainingPipeline.Fit(trainingDataView);

            Console.WriteLine("=============== End of training process ===============");
            return model;
        }

        private static void Evaluate(MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> trainingPipeline)
        {
            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "evaluation");
            PrintRegressionFoldsAverageMetrics(crossValidationResults);
        }

        private static void SaveModel(MLContext mlContext, ITransformer mlModel, string modelRelativePath, DataViewSchema modelInputSchema)
        {
            Console.WriteLine($"=============== Saving the model  ===============");
            mlContext.Model.Save(mlModel, modelInputSchema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public static void PrintRegressionMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn:        {metrics.LossFunction:0.##}");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        public static void PrintRegressionFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            Console.WriteLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            Console.WriteLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
        }
    }
}
