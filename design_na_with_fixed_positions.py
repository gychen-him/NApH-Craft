#!/usr/bin/env python
"""
NA-MPNN Nucleic Acid Design Script with Fixed Positions

This script performs inverse folding on nucleic acid structures using NA-MPNN,
allowing you to fix specific positions while redesigning others.

Usage:
    python design_na_with_fixed_positions.py \
        --cif_path /path/to/structure.cif \
        --out_folder ./output \
        --fixed_positions "B1 B2 B3" \
        --num_sequences 10 \
        --temperature 0.1

Arguments:
    --cif_path: Path to the input CIF/PDB file
    --out_folder: Output directory for designed sequences
    --fixed_positions: Space-separated list of positions to keep fixed (e.g., "B1 B2 B3")
                       Format: ChainID + ResidueNumber (e.g., B1 means chain B, residue 1)
    --redesigned_positions: (Optional) Only redesign these positions, fix everything else
    --chains_to_design: (Optional) Only design these chains (e.g., "B" for DNA chain)
    --design_na_only: (Optional) If 1, only design nucleic acid chains (default: 1)
    --num_sequences: Number of sequences to generate (default: 10)
    --temperature: Sampling temperature (default: 0.1, lower = more conservative)
    --batch_size: Batch size for generation (default: 1)
"""

import argparse
import os
import sys
import json
import zipfile
import tempfile
import shutil

# Add the inference directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
inference_dir = os.path.join(script_dir, "inference")
sys.path.insert(0, inference_dir)
sys.path.insert(0, script_dir)


def extract_cif_from_zip(zip_path, output_dir, model_index=0):
    """Extract a CIF file from an AlphaFold zip archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        cif_files = [f for f in z.namelist() if f.endswith('.cif') and f'model_{model_index}' in f]
        if not cif_files:
            # Try without model index
            cif_files = [f for f in z.namelist() if f.endswith('.cif')]
        
        if not cif_files:
            raise ValueError(f"No CIF files found in {zip_path}")
        
        cif_file = cif_files[0]
        z.extract(cif_file, output_dir)
        return os.path.join(output_dir, cif_file)


def main(args):
    from prody import writePDB
    import torch
    import random
    import numpy as np
    from data_utils import make_pair_bias, get_seq_rec, get_score, parse_PDB, featurize
    from model_utils import ProteinMPNN

    # Model configuration for NA-MPNN
    atom_types = [
        'N', 'CA', 'C', 'O',  # protein atoms
        'OP1', 'OP2', 'P', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"  # nucleic acid atoms
    ]
    atom_dict = dict(zip(atom_types, range(len(atom_types))))

    polytypes = ['PP', 'DNA', 'RNA', 'UNK', 'MAS', 'PAD']
    polytype_to_int = dict(zip(polytypes, range(len(polytypes))))

    restypes = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'UNK', 'DA', 'DC', 'DG', 'DT', 'DX', 'A', 'C', 'G', 'U', 'RX', 'MAS', 'PAD'
    ]

    restype_3_to_1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'UNK': 'X',
        'DA': 'a', 'DC': 'c', 'DG': 'g', 'DT': 't', 'DX': 'x',
        'A': 'b', 'C': 'd', 'G': 'h', 'U': 'u', 'RX': 'y',
        'MAS': '-', 'PAD': '+'
    }

    restype_to_int = dict(zip(restypes, range(len(restypes))))
    int_to_restype = dict(zip(range(len(restypes)), restypes))
    alphabet = [restype_3_to_1[int_to_restype[i]] for i in range(len(int_to_restype))]
    restype_1to3 = {v: k for k, v in restype_3_to_1.items()}

    # Use shared tokens for DNA and RNA
    dna_char_to_rna_char = {}
    if args.na_shared_tokens:
        restype_to_int["A"] = restype_to_int["DA"]
        restype_to_int["C"] = restype_to_int["DC"]
        restype_to_int["G"] = restype_to_int["DG"]
        restype_to_int["U"] = restype_to_int["DT"]
        restype_to_int["RX"] = restype_to_int["DX"]

        dna_char_to_rna_char[restype_3_to_1["DA"]] = restype_3_to_1["A"]
        dna_char_to_rna_char[restype_3_to_1["DC"]] = restype_3_to_1["C"]
        dna_char_to_rna_char[restype_3_to_1["DG"]] = restype_3_to_1["G"]
        dna_char_to_rna_char[restype_3_to_1["DT"]] = restype_3_to_1["U"]
        dna_char_to_rna_char[restype_3_to_1["DX"]] = restype_3_to_1["RX"]

    restype_STRtoINT = {restype_3_to_1[k]: v for k, v in restype_to_int.items()}
    restype_INTtoSTR = {}
    for k, v in restype_STRtoINT.items():
        if v not in restype_INTtoSTR:
            restype_INTtoSTR[v] = k

    vocab = 33
    num_letters = 33

    # Set random seed
    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output folders
    base_folder = args.out_folder
    if base_folder[-1] != '/':
        base_folder = base_folder + '/'
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(base_folder + 'seqs', exist_ok=True)
    if args.output_pdbs:
        os.makedirs(base_folder + 'backbones', exist_ok=True)

    # Load model
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(script_dir, "models/design_model/s_19137.pt")
    
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    k_neighbors = 32
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        model_type="na_mpnn",
        vocab=vocab,
        num_letters=num_letters,
        atom_dict=atom_dict,
        restype_to_int=restype_to_int,
        polytype_to_int=polytype_to_int
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Prepare input structure
    input_path = args.cif_path
    temp_dir = None
    
    # Handle zip files
    if input_path.endswith('.zip'):
        temp_dir = tempfile.mkdtemp()
        input_path = extract_cif_from_zip(input_path, temp_dir, args.model_index)
        print(f"Extracted CIF file: {input_path}")

    # Get structure name
    name = os.path.basename(input_path)
    if name.endswith('.pdb') or name.endswith('.cif'):
        name = name[:-4]

    # Set up amino acid bias (omit protein residues if only designing NA)
    bias_AA = torch.zeros([num_letters], device=device, dtype=torch.float32)
    
    # Omit amino acids list
    omit_AA_list = args.omit_AA
    if args.na_shared_tokens:
        omit_AA_list = omit_AA_list + "bdhuy"  # Omit separate RNA tokens
    
    omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32), 
        device=device
    )

    # Parse structure
    print(f"Parsing structure: {input_path}")
    macromolecule_dict, backbone, other_atoms, icodes, water_atoms = parse_PDB(
        input_path,
        device=device,
        chains=args.parse_these_chains_only,
        model_type="na_mpnn",
        parse_na_only=args.parse_na_only,
        na_shared_tokens=args.na_shared_tokens,
        load_residues_with_missing_atoms=args.load_residues_with_missing_atoms
    )

    # Create residue encoding mapping
    R_idx_list = list(macromolecule_dict["R_idx"].cpu().numpy())
    chain_letters_list = list(macromolecule_dict["chain_letters"])
    encoded_residues = []
    for i in range(len(R_idx_list)):
        tmp = str(chain_letters_list[i]) + str(R_idx_list[i]) + icodes[i]
        encoded_residues.append(tmp)
    encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))

    # Print structure info
    print(f"\n=== Structure Information ===")
    print(f"Total residues: {len(encoded_residues)}")
    unique_chains = list(set(chain_letters_list))
    for chain in sorted(unique_chains):
        chain_residues = [r for r, c in zip(encoded_residues, chain_letters_list) if c == chain]
        print(f"Chain {chain}: {len(chain_residues)} residues ({chain_residues[0]} - {chain_residues[-1]})")
    
    na_chains = macromolecule_dict["na_chain_letters"]
    print(f"Nucleic acid chains: {list(set(na_chains))}")

    # Parse fixed positions
    fixed_residues = []
    if args.fixed_positions:
        fixed_residues = [item.strip() for item in args.fixed_positions.split()]
        print(f"\nFixed positions: {fixed_residues}")
    
    # Create fixed positions mask (1 = design, 0 = fixed)
    fixed_positions = torch.tensor(
        [int(item not in fixed_residues) for item in encoded_residues], 
        device=device
    )

    # Parse redesigned positions (if specified, only redesign these)
    if args.redesigned_positions:
        redesigned_residues = [item.strip() for item in args.redesigned_positions.split()]
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues], 
            device=device
        )
        print(f"Redesigned positions: {redesigned_residues}")
    else:
        redesigned_positions = torch.zeros_like(fixed_positions)

    # Determine chains to design
    if args.chains_to_design:
        chains_to_design_list = args.chains_to_design.split(",")
    else:
        chains_to_design_list = list(set(chain_letters_list))

    if args.design_na_only:
        chains_to_design_list = [c for c in chains_to_design_list if c in na_chains]
        print(f"Designing only NA chains: {chains_to_design_list}")

    chain_mask = torch.tensor(
        np.array([item in chains_to_design_list for item in chain_letters_list], dtype=np.int32), 
        device=device
    )

    # Final chain mask: chain_mask * fixed_positions * (1 - redesigned_positions)
    macromolecule_dict["chain_mask"] = chain_mask * fixed_positions * (1 - redesigned_positions)

    # Print design summary
    design_mask = macromolecule_dict["chain_mask"].cpu().numpy()
    designed_positions = [r for r, m in zip(encoded_residues, design_mask) if m == 1]
    fixed_positions_list = [r for r, m in zip(encoded_residues, design_mask) if m == 0]
    
    print(f"\n=== Design Summary ===")
    print(f"Positions to design: {len(designed_positions)}")
    print(f"Fixed positions: {len(fixed_positions_list)}")
    
    # Show original sequence for designed region
    S_original = macromolecule_dict["S"].cpu().numpy()
    original_seq = ""
    for i, (res, m) in enumerate(zip(encoded_residues, design_mask)):
        if chain_letters_list[i] in chains_to_design_list:
            aa = restype_INTtoSTR[S_original[i]]
            if m == 1:
                original_seq += aa.upper()  # Designed positions in uppercase
            else:
                original_seq += aa.lower()  # Fixed positions in lowercase
    
    print(f"\nOriginal NA sequence (UPPERCASE=to design, lowercase=fixed):")
    print(f"{original_seq}")

    # Symmetry settings (optional)
    remapped_symmetry_residues = [[]]
    symmetry_weights = [[]]

    # Run inference
    print(f"\n=== Running NA-MPNN ===")
    print(f"Temperature: {args.temperature}")
    print(f"Number of sequences: {args.num_sequences}")
    print(f"Batch size: {args.batch_size}")

    with torch.no_grad():
        feature_dict = featurize(macromolecule_dict)
        feature_dict["batch_size"] = args.batch_size
        B, L, _, _ = feature_dict["X"].shape

        feature_dict["temperature"] = args.temperature
        feature_dict["bias"] = (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
        feature_dict["symmetry_residues"] = remapped_symmetry_residues
        feature_dict["symmetry_weights"] = symmetry_weights

        sampling_probs_list = []
        log_probs_list = []
        S_list = []
        loss_list = []
        loss_per_residue_list = []

        num_batches = (args.num_sequences + args.batch_size - 1) // args.batch_size
        
        for batch_idx in range(num_batches):
            feature_dict["randn"] = torch.randn(
                [feature_dict["batch_size"], feature_dict["mask"].shape[1]], 
                device=device
            )
            
            output_dict = model.sample(feature_dict)
            
            loss, loss_per_residue = get_score(
                output_dict["S"], output_dict["log_probs"], 
                feature_dict["mask"] * feature_dict["chain_mask"], 
                num_letters
            )

            S_list.append(output_dict["S"])
            log_probs_list.append(output_dict["log_probs"])
            sampling_probs_list.append(output_dict["sampling_probs"])
            loss_list.append(loss)
            loss_per_residue_list.append(loss_per_residue)

        S_stack = torch.cat(S_list, 0)[:args.num_sequences]
        log_probs_stack = torch.cat(log_probs_list, 0)[:args.num_sequences]
        sampling_probs_stack = torch.cat(sampling_probs_list, 0)[:args.num_sequences]
        loss_stack = torch.cat(loss_list, 0)[:args.num_sequences]
        loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)[:args.num_sequences]

        rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
        rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

        # Create native sequence string
        native_seq_list = []
        for i, AA in enumerate(feature_dict["S"][0].cpu().numpy()):
            if feature_dict["rna_mask_for_token_conversion"][0, i] == 1:
                native_seq_list.append(dna_char_to_rna_char.get(restype_INTtoSTR[AA], restype_INTtoSTR[AA]))
            else:
                native_seq_list.append(restype_INTtoSTR[AA])
        native_seq = "".join(native_seq_list)
        
        seq_np = np.array(list(native_seq))
        seq_out_str = []
        for mask in macromolecule_dict['mask_c']:
            seq_out_str += list(seq_np[mask.cpu().numpy()])
            seq_out_str += ['/']
        seq_out_str = "".join(seq_out_str)[:-1]

        # Write outputs
        output_fasta = base_folder + 'seqs/' + name + '.fa'
        
        fasta_entries = []
        fasta_entries.append(
            '>{}, T={}, seed={}, num_designed_res={}, model_path={}\n{}'.format(
                name, args.temperature, seed, 
                int(torch.sum(rec_mask).cpu().numpy()),
                checkpoint_path, seq_out_str
            )
        )

        print(f"\n=== Generated Sequences ===")
        for ix in range(S_stack.shape[0]):
            seq_rec_print = np.format_float_positional(rec_stack[ix].cpu().numpy(), unique=False, precision=4)
            loss_np = np.format_float_positional(np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4)

            seq_list = []
            for i, AA in enumerate(S_stack[ix].cpu().numpy()):
                if feature_dict["rna_mask_for_token_conversion"][0, i] == 1:
                    seq_list.append(dna_char_to_rna_char.get(restype_INTtoSTR[AA], restype_INTtoSTR[AA]))
                else:
                    seq_list.append(restype_INTtoSTR[AA])
            seq = "".join(seq_list)

            # Output PDB if requested
            if args.output_pdbs:
                seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])
                bfactor_prody = loss_per_residue_stack[ix].cpu().numpy()
                
                for i, (chain_letter, res_idx) in enumerate(zip(chain_letters_list, R_idx_list)):
                    residue = backbone.select("chain {} and resnum {}".format(chain_letter, res_idx))
                    if residue is not None:
                        residue.setResnames(seq_prody[i])
                        residue.setBetas(np.exp(-bfactor_prody[i]) * (bfactor_prody[i] > 0.01).astype(np.float32))

                output_backbone_path = base_folder + 'backbones/' + name + '_' + str(ix + 1) + ".pdb"
                if other_atoms:
                    writePDB(output_backbone_path, backbone + other_atoms)
                else:
                    writePDB(output_backbone_path, backbone)

            # Create fasta entry
            seq_np = np.array(list(seq))
            seq_out_str = []
            for mask in macromolecule_dict['mask_c']:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += ['/']
            seq_out_str = "".join(seq_out_str)[:-1]

            # Extract only NA sequence for display
            na_seq = ""
            for i, c in enumerate(chain_letters_list):
                if c in chains_to_design_list:
                    na_seq += seq_list[i]
            
            print(f"Seq {ix + 1}: {na_seq} (confidence={loss_np}, seq_rec={seq_rec_print})")

            fasta_entries.append(
                '>{}, id={}, T={}, seed={}, confidence={}, seq_rec={}\n{}'.format(
                    name, ix + 1, args.temperature, seed, loss_np, seq_rec_print, seq_out_str
                )
            )

        # Write fasta file
        with open(output_fasta, 'w') as f:
            f.write("\n".join(fasta_entries))
        
        print(f"\n=== Output Files ===")
        print(f"Sequences saved to: {output_fasta}")
        if args.output_pdbs:
            print(f"PDB backbones saved to: {base_folder}backbones/")

    # Cleanup temp directory
    if temp_dir:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NA-MPNN nucleic acid design with fixed positions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument("--cif_path", type=str, required=True,
                        help="Path to input CIF/PDB file or ZIP archive")
    parser.add_argument("--out_folder", type=str, required=True,
                        help="Output folder for designed sequences")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to model checkpoint (default: models/design_model/s_19137.pt)")
    
    # Design parameters
    parser.add_argument("--fixed_positions", type=str, default="",
                        help="Space-separated list of positions to keep fixed (e.g., 'B1 B2 B3')")
    parser.add_argument("--redesigned_positions", type=str, default="",
                        help="Only redesign these positions (everything else fixed)")
    parser.add_argument("--chains_to_design", type=str, default=None,
                        help="Comma-separated chains to design (e.g., 'B')")
    parser.add_argument("--design_na_only", type=int, default=1,
                        help="1 = only design nucleic acid chains")
    parser.add_argument("--omit_AA", type=str, default="ARNDCQEGHILKMFPSTWYVX",
                        help="Amino acids to omit from design (default: all protein AAs)")
    
    # Sampling parameters
    parser.add_argument("--num_sequences", type=int, default=10,
                        help="Number of sequences to generate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (lower = more conservative)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    # Structure parsing
    parser.add_argument("--model_index", type=int, default=0,
                        help="Model index to extract from ZIP file")
    parser.add_argument("--parse_these_chains_only", type=str, default="",
                        help="Only parse these chains (e.g., 'AB')")
    parser.add_argument("--parse_na_only", type=int, default=0,
                        help="1 = parse only nucleic acid chains")
    parser.add_argument("--na_shared_tokens", type=int, default=1,
                        help="1 = use shared tokens for DNA and RNA")
    parser.add_argument("--load_residues_with_missing_atoms", type=int, default=0,
                        help="1 = load residues with missing atoms")
    
    # Output options
    parser.add_argument("--output_pdbs", type=int, default=1,
                        help="1 = output PDB files with designed sequences")
    
    args = parser.parse_args()
    main(args)

