import argparse
import sys
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['cnn', 'gan'], required=True, help='Select model: cnn or gan')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--train_mode', type=str, choices=['normal', 'fine_tune'], default='normal', help='Training mode: normal or fine_tune (for train mode only)')
    parser.add_argument('--data_dir', type=str, default='data/sample_data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes (for cnn)')
    parser.add_argument('--img_size', type=int, default=64, help='Image size')
    parser.add_argument('--input_model', type=str, default=None, help='Path to input model checkpoint for CNN (required for fine tuning)')
    parser.add_argument('--input_model_G', type=str, default=None, help='Path to GAN generator checkpoint (required for fine tuning)')
    parser.add_argument('--input_model_D', type=str, default=None, help='Path to GAN discriminator checkpoint (required for fine tuning)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    trainer = None
    if args.model == 'cnn':
        from trainer.cnn_trainer import CNNTrainer
        trainer = CNNTrainer(data_dir=args.data_dir, num_classes=args.num_classes, batch_size=args.batch_size, img_size=args.img_size)
    elif args.model == 'gan':
        from trainer.gan_trainer import GANTrainer
        trainer = GANTrainer(data_dir=args.data_dir, batch_size=args.batch_size, image_size=args.img_size)
    else:
        print("Invalid model selection.")
        sys.exit(1)
    
    # For training, if an input model is supplied, load it.
    if args.mode == 'train':
        # First check if an input model is supplied for training.
        if args.model == 'cnn':
            if args.train_mode == 'fine_tune':
                if args.input_model is None:
                    print("Error: Fine tuning mode for CNN requires an input model checkpoint. Please supply --input_model.")
                    sys.exit(1)
                else:
                    trainer.load_checkpoint(args.input_model)
            else:
                if args.input_model is not None:
                    trainer.load_checkpoint(args.input_model)
        elif args.model == 'gan':
            if args.train_mode == 'fine_tune':
                if args.input_model_G is None or args.input_model_D is None:
                    print("Error: Fine tuning mode for GAN requires input checkpoints for both generator and discriminator. Please supply --input_model_G and --input_model_D.")
                    sys.exit(1)
                else:
                    trainer.load_checkpoint(args.input_model_G, args.input_model_D)
            else:
                if args.input_model_G is not None and args.input_model_D is not None:
                    trainer.load_checkpoint(args.input_model_G, args.input_model_D)
        # Start training
        trainer.train(num_epochs=args.epochs, mode=args.train_mode)
        # Also test the model after training to ensure that it is generalizing well
        trainer.test()
        # Save the model
        if args.model == 'cnn':
            trainer.save('cnn_model.pth')
        else:
            trainer.save('gan_generator.pth', 'gan_discriminator.pth')
    elif args.mode == 'test':
        # Ensure that the model is loaded for testing
        if args.model == 'cnn':
            if args.input_model is None:
                print("Error: Testing mode for CNN requires an input model checkpoint. Please supply --input_model.")
                sys.exit(1)
            else:
                trainer.load_checkpoint(args.input_model)
        elif args.model == 'gan':
            if args.input_model_G is None or args.input_model_D is None:
                print("Error: Testing mode for GAN requires input checkpoints for both generator and discriminator. Please supply --input_model_G and --input_model_D.")
                sys.exit(1)
            else:
                trainer.load_checkpoint(args.input_model_G, args.input_model_D)
        # Start testing
        trainer.test()
    else:
        print("Invalid mode.")
        sys.exit(1)

if __name__ == '__main__':
    main()
