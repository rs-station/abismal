FILES=`ls reflection_data/46294233/out/prep_0F_spread_1209_1B_*.{expt,refl}`
#FILES=`ls reflection_data/46294233/out/prep_0F_spread_1209_1B_*.{expt,refl}|head -4`
OUTDIR=results/scratch16

PARAMS=(
    #--debug
    #--unity-posterior #for debugging -- don't do any spread
    --epochs 100
    --d-model 32
    #--wavelength-range 1.87 1.91
    --layers 20
    --steps-per-epoch 1_000
    --num-cpus 10
    --batch-size 100
    --kl-weight=1e-2
    --scale-kl-weight=1e0
    --charge=3
    #--studentt-dof=32
    --keras-verbosity 1
    #--isigi-cutoff=0.0
    --dmin 2.5
    --model-file reference_data/L10198_allcombined_OEC_waters_t3a_1.90_118_45.pdb
    --element Mn
    --out-dir $OUTDIR
    --epsilon=1e-12
    #--optimizer=Adam
    --optimizer=AdaBelief
)

# Prep the output directory
mkdir -p $OUTDIR
cp $0 $OUTDIR/spread.sh

# Run spread calculation
abismal.spread ${PARAMS[@]} $FILES
