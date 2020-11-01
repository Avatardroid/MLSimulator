using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;


namespace WindowsFormsApp3
{
    public partial class Form1 : Form
    {

        public Form1()
        {
            InitializeComponent();

        }

        //string image1 = null;
        //string image2 = null;

        string GetFolderPath(string Description)
        {
            FolderBrowserDialog folderBrowser = new FolderBrowserDialog();
            folderBrowser.Description = Description; 
            if (folderBrowser.ShowDialog() == DialogResult.OK)
            {
                return folderBrowser.SelectedPath;
            }
            else
            {
                return string.Empty;
            }
        }

        string GetFilePath(string Description)
        {
            FolderBrowserDialog folderBrowser = new FolderBrowserDialog();
            folderBrowser.Description = Description;
            if (folderBrowser.ShowDialog() == DialogResult.OK)
            {
                return folderBrowser.SelectedPath;
            }
            else
            {
                return string.Empty;
            }
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            Directory.CreateDirectory("metrics");
            Directory.CreateDirectory("models");
            Directory.CreateDirectory("out");
        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog fileBrowser = new OpenFileDialog();
            fileBrowser.Title = "Выберите файл с метриками";
            fileBrowser.InitialDirectory = Path.Combine(Environment.CurrentDirectory,"metrics");
            if (fileBrowser.ShowDialog() == DialogResult.OK)
            {
                ModelBuilder.CreateModel(fileBrowser.FileName);
                MessageBox.Show("Модель обучена");
            }
        }


        public float[] GetSquare(string image1, string image2)
        {
            Image<Bgr, Byte> RefImg = null;
            Image<Bgr, Byte> DiffBGRefImg = null;
            Image<Gray, Byte> DiffBGRefImgGray = null;
            Image<Gray, Byte> RefImgGray = null;

            Image<Bgr, Byte> CompImg = null;
            Image<Bgr, Byte> DiffBGCompImg = null;
            Image<Gray, Byte> DiffBGCompImgGray = null;
            Image<Gray, Byte> CompImgGray = null;

            float ExpertPix = 0;
            float ExcessivePix = 0;
            float MissingPix = 0;
            float JointPix = 0;
            float SamplePix = 0;
            float BackgroundRefPix = 0;
            float BackgroundCompPix = 0;

            if (image1 != string.Empty)
            {
                RefImg = new Bitmap(image1).ToImage<Bgr, Byte>();
                DiffBGRefImg = ChangeBackground(new Bitmap(image1).ToImage<Bgr, Byte>());
                DiffBGRefImgGray = DiffBGRefImg.Convert<Gray, byte>();
                RefImgGray = RefImg.Convert<Gray, byte>();

                //Image<Bgr, byte> result_Expert = RefImg.CopyBlank();
                Image<Gray, Byte> MaskDifferenceExpert = RefImgGray.Cmp(DiffBGRefImgGray, CmpType.GreaterEqual);
                //result_Expert.SetValue(new Bgr(Color.Azure), MaskDifferenceExpert);
                ExpertPix = MaskDifferenceExpert.Data.Cast<byte>().ToList().Where(x => x != 0).Count();
                BackgroundRefPix = RefImg.Bytes.Length - ExpertPix;
            }
            if (image2 != string.Empty)
            {
                CompImg = new Bitmap(image2).ToImage<Bgr, Byte>();
                DiffBGCompImg = ChangeBackground(new Bitmap(image2).ToImage<Bgr, Byte>());
                DiffBGCompImgGray = DiffBGCompImg.Convert<Gray, byte>();
                CompImgGray = CompImg.Convert<Gray, byte>();

                //Image<Bgr, byte> result_Sample = CompImg.CopyBlank();
                Image<Gray, Byte> MaskDifferenceSample = CompImgGray.Cmp(DiffBGCompImgGray, CmpType.GreaterEqual);
                //result_Sample.SetValue(new Bgr(Color.Azure), MaskDifferenceSample);
                SamplePix = MaskDifferenceSample.Data.Cast<byte>().ToList().Where(x => x != 0).Count();
                BackgroundCompPix = CompImg.Bytes.Length - ExpertPix;
            }
            if (image2 != string.Empty && image2 != string.Empty)
            {
                //Image<Bgr, byte> result_Excessive = RefImg.CopyBlank();
                Image<Gray, Byte> MaskDifferenceExcessive = RefImgGray.Cmp(CompImgGray, CmpType.LessThan);
                //result_Excessive.SetValue(new Bgr(Color.Blue), MaskDifferenceExcessive);
                //pictureBox5.Image = result_Excessive.ToBitmap();
                ExcessivePix = MaskDifferenceExcessive.Data.Cast<byte>().ToList().Where(x => x != 0).Count();

                //Image<Bgr, byte> result_Missing = RefImg.CopyBlank();
                Image<Gray, Byte> MaskDifferenceMissing = RefImgGray.Cmp(CompImgGray, CmpType.GreaterThan);
                //result_Missing.SetValue(new Bgr(Color.Aqua), MaskDifferenceMissing);
                //pictureBox6.Image = result_Missing.ToBitmap();
                MissingPix = MaskDifferenceMissing.Data.Cast<byte>().ToList().Where(x => x != 0).Count();

                //Image<Bgr, byte> resultJoint = CompImg.CopyBlank();
                Image<Gray, Byte> MaskDifferenceJoint = DiffBGCompImgGray.Cmp(RefImgGray, CmpType.Equal);
                //resultJoint.SetValue(new Bgr(Color.Red), MaskDifferenceJoint);
                JointPix = MaskDifferenceJoint.Data.Cast<byte>().ToList().Where(x => x != 0).Count();
            }
            //decimal BackgroundCompPix = CompImg.Bytes.Length - (ExcessivePix + SamplePix);
            return new float[] { ExpertPix, BackgroundRefPix, SamplePix, BackgroundCompPix, ExcessivePix, MissingPix, JointPix};
        }
        private void button3_Click(object sender, EventArgs e)
        {
            string work_folder = GetFolderPath("Выберите папку с DataSet");
            if (work_folder == string.Empty)
            {
                return;
            }
            List<string[]> ml_info = new List<string[]>();
            List<string[]> list_oc = new List<string[]>();
            ml_info.Add(new string[] {"pic","evaluation", "expert_square", "background_expert", "sample_square", "background_sample", "excessive_square", "missing_square", "joint_square" });
            using (StreamReader sr = new StreamReader(Path.Combine(work_folder, @"OpenPart.csv")))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    list_oc.Add(line.Split(','));
                }
            }
            toolStripProgressBar1.Maximum = list_oc.Count();
            int samples_count = new DirectoryInfo(Path.Combine(work_folder)).GetDirectories("sample*").Count();
            for (int i = 1; i < list_oc.Count(); i++)
            {
                
                for (int k = 1; k <= samples_count; k++)
                {
                    float[] temp = GetSquare(new DirectoryInfo(Path.Combine(work_folder, @"Expert")).GetFiles($"{list_oc[i][0].Split('.')[0]}*")[0].FullName, new DirectoryInfo(Path.Combine(work_folder, $@"sample_{k}")).GetFiles($"{list_oc[i][0].Split('.')[0]}*")[0].FullName);
                    ml_info.Add(new string[] { list_oc[i][0], list_oc[i][k], temp[0].ToString(), temp[1].ToString() , temp[2].ToString() , temp[3].ToString() , temp[4].ToString() , temp[5].ToString() , temp[6].ToString() });
                }
                toolStripProgressBar1.Value++;
                Application.DoEvents();
            }

            toolStripProgressBar1.Value++;
            Application.DoEvents();
            using (StreamWriter sr = new StreamWriter(Path.Combine(Path.Combine(Environment.CurrentDirectory, "metrics"), $@"MLMetrics_{DateTime.Now.ToString($"HHmmss_ddMMyy")}.csv")))
            {
                foreach(string[] temp_line in ml_info)
                {
                    sr.WriteLine(String.Join(";", temp_line));
                }
            }
            toolStripProgressBar1.Value = 0;
            MessageBox.Show("Метрики для обучения модели созданы");
        }


        public static Image<Bgr, byte> ChangeBackground(Image<Bgr, byte> rgbimage)
        {
            Image<Bgr, byte> ret = rgbimage;
            var image = rgbimage.InRange(new Bgr(0, 0, 0), new Bgr(1, 1, 1));
            var mat = rgbimage.Mat;
            mat.SetTo(new MCvScalar(255, 0, 0), image);
            mat.CopyTo(ret);
            return ret;
        }

        string work_folder_view = string.Empty;
        private void button4_Click(object sender, EventArgs e)
        {
            listBox1.Items.Clear();
            work_folder_view = GetFolderPath("Выберите папку с DataSet");
            if (work_folder_view == string.Empty)
            {
                return;
            }
            OpenFileDialog fileBrowser = new OpenFileDialog();
            fileBrowser.Title = "Выберите файл с обученой моделью";
            fileBrowser.InitialDirectory = Path.Combine(Environment.CurrentDirectory, "models");
            if (fileBrowser.ShowDialog() == DialogResult.OK)
            {
                ConsumeModel.PATH_MODEL = fileBrowser.FileName;
            }
            else
            {
                return;
            }
            FileInfo[] expert_files = new DirectoryInfo(Path.Combine(work_folder_view, @"Expert")).GetFiles();
            foreach (FileInfo fi in expert_files)
            {
                listBox1.Items.Add($"{fi.Name.Split('_')[0]}_{fi.Name.Split('_')[1]}");
            }
        }
        private void listBox1_DoubleClick(object sender, EventArgs e)
        {
            listBox1.Enabled = false;
            string evaluation = string.Empty;
            string sel_item = listBox1.SelectedItem.ToString();
            pictureBox1.Image = new Bitmap(new DirectoryInfo(Path.Combine(work_folder_view, @"Expert")).GetFiles($"{sel_item}*")[0].FullName);
            pictureBox2.Image = new Bitmap(new DirectoryInfo(Path.Combine(work_folder_view, @"Origin")).GetFiles($"{sel_item}*")[0].FullName);
            pictureBox4.Image = new Bitmap(new DirectoryInfo(Path.Combine(work_folder_view, @"sample_1")).GetFiles($"{sel_item}*")[0].FullName);
            pictureBox5.Image = new Bitmap(new DirectoryInfo(Path.Combine(work_folder_view, @"sample_2")).GetFiles($"{sel_item}*")[0].FullName);
            pictureBox6.Image = new Bitmap(new DirectoryInfo(Path.Combine(work_folder_view, @"sample_3")).GetFiles($"{sel_item}*")[0].FullName);
            for (int k = 1; k <= 3; k++)
            {
                float[] temp = GetSquare(new DirectoryInfo(Path.Combine(work_folder_view, @"Expert")).GetFiles($"{sel_item}*")[0].FullName, new DirectoryInfo(Path.Combine(work_folder_view, $@"sample_{k}")).GetFiles($"{sel_item}*")[0].FullName);
                ModelInput sampleData = new ModelInput()
                {
                    Expert_square = temp[0],
                    Background_expert = temp[1],
                    Sample_square = temp[2],
                    Background_sample = temp[3],
                    Excessive_square = temp[4],
                    Missing_square = temp[5],
                    Joint_square = temp[6],
                };
                evaluation += $"{ConsumeModel.Predict(sampleData).Score} ";
            }
            label2.Text = evaluation;
            listBox1.Enabled = true;
        }

        private void button5_Click(object sender, EventArgs e)
        {
            string work_folder = GetFolderPath("Выберите папку с DataSet");
            if (work_folder == string.Empty)
            {
                return;
            }
            List<string[]> list_oc = new List<string[]>();
            using (StreamReader sr = new StreamReader(Path.Combine(work_folder, @"OpenPart.csv")))
            {
                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    list_oc.Add(line.Split(','));
                }
            }
            OpenFileDialog fileBrowser = new OpenFileDialog();
            fileBrowser.Title = "Выберите файл с обученой моделью";
            fileBrowser.InitialDirectory = Path.Combine(Environment.CurrentDirectory, "models");
            if (fileBrowser.ShowDialog() == DialogResult.OK)
            {
                ConsumeModel.PATH_MODEL = fileBrowser.FileName;
            }
            else
            {
                return;
            }
            string file_date = DateTime.Now.ToString($"HHmmss_ddMMyy");
            FileInfo[] expert_files = new DirectoryInfo(Path.Combine(work_folder, @"Expert")).GetFiles();
            toolStripProgressBar1.Maximum = expert_files.Count();
            using (StreamWriter sr = new StreamWriter(Path.Combine("out", $@"evaluation_result_{file_date}.csv"), true))
            {
                sr.WriteLine(String.Join(",", new string[] { "Case", "Sample 1", "Sample 2", "Sample 3" }));
            }
            for (int i = 1; i < expert_files.Count(); i++)
            {
                string[] temp_oc = new string[4];
                string find_string = $"{expert_files[i].Name.Split('_')[0]}_{expert_files[i].Name.Split('_')[1]}.png";
                int samples_count = new DirectoryInfo(Path.Combine(work_folder)).GetDirectories("sample*").Count();
                if (list_oc.Select(x => x[0]).ToList().FindIndex(y => y == find_string) == -1)
                {
                    for (int k = 1; k <= samples_count; k++)
                    {
                        float[] temp = GetSquare(new DirectoryInfo(Path.Combine(work_folder, @"Expert")).GetFiles($"{expert_files[i].Name.Split('_')[0]}_{expert_files[i].Name.Split('_')[1]}*")[0].FullName, new DirectoryInfo(Path.Combine(work_folder, $@"sample_{k}")).GetFiles($"{expert_files[i].Name.Split('_')[0]}_{expert_files[i].Name.Split('_')[1]}*")[0].FullName);
                        ModelInput sampleData = new ModelInput()
                        {
                            Expert_square = temp[0],
                            Background_expert = temp[1],
                            Sample_square = temp[2],
                            Background_sample = temp[3],
                            Excessive_square = temp[4],
                            Missing_square = temp[5],
                            Joint_square = temp[6]
                        };
                        float evaluation = (float)ConsumeModel.Predict(sampleData).Score;
                        temp_oc[k] = Math.Round(evaluation).ToString();
                    }
                    temp_oc[0] = $"{expert_files[i].Name.Split('_')[0]}_{expert_files[i].Name.Split('_')[1]}.png";
                    using (StreamWriter sr = new StreamWriter(Path.Combine("out", $@"evaluation_result_{file_date}.csv"), true))
                    {
                        sr.WriteLine(String.Join(",", temp_oc));
                    }
                }
                toolStripProgressBar1.Value++;
            }
            toolStripProgressBar1.Value++;
            MessageBox.Show("Файл c оценками создан");
            toolStripProgressBar1.Value = 0;
        }


    }
}
