# face
python main.py -i clip_face -s clip_ddim --doc celeba_hq --timesteps 100 --seed 1234 --model_type "face" --prompt "blonde beauty" --batch_size 1
python main.py -i seg_face -s parse_ddim --doc celeba_hq --timesteps 100 --rho_scale 0.2 --seed 1234 --stop 200 --ref_path ./images/294.jpg --batch_size 1
python main.py -i sketch_face -s sketch_ddim --doc celeba_hq --timesteps 100 --rho_scale 20.0 --seed 1234 --stop 100 --ref_path ./images/294.jpg --batch_size 1
python main.py -i landmark_face -s land_ddim --doc celeba_hq --timesteps 100 --rho_scale 500.0 --seed 1234 --stop 200 --ref_path ./images/2334.jpg --batch_size 1
python main.py -i arcface_face -s arc_ddim --doc celeba_hq --timesteps 100 --rho_scale 100.0 --seed 1234 --stop 100 --ref_path  ./images/id10.png --batch_size 1

# imagenet
python main.py -i clip_imagenet -s clip_ddim --doc imagenet --timesteps 100 --seed 1234 --model_type "imagenet" --prompt "orange" --batch_size 1
