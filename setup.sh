cd $(dirname $(realpath ${0}))

mkdir anaconda
mkdir transformers

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b -f -p anaconda
rm Miniconda3-latest-Linux-x86_64.sh

source anaconda/bin/activate anaconda/
conda install -y python=3.9
conda install -y -c nvidia -c pytorch pytorch=1.13.0 pytorch-cuda=11.6 torchvision=0.14.0
pip install numpy==1.23.5
pip install Pillow==9.3.0
pip install PyYAML==6.0
pip install tqdm==4.64.1
pip install transformers==4.25.1

if [ ${1} = codraw ]; then
  wget https://cmg-datasets.s3.us-east-2.amazonaws.com/codraw/codraw_data.zip
  unzip codraw_data.zip
  rm codraw_data.zip

elif [ ${1} = iclevr ]; then
  wget https://cmg-datasets.s3.us-east-2.amazonaws.com/iclevr/iclevr_data.zip
  unzip iclevr_data.zip
  rm iclevr_data.zip

else
  exit 1
fi
