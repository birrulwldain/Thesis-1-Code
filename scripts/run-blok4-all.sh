for file in raw/Skala-5/0/S*-D0.5us*.asc; do
    echo -e "\n============================================="
    echo "MEMPROSES FILE: $file"
    python empirical_validation.py --model model_inversi_svr.pkl --csv "$file"
done
