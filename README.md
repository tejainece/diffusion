# Diffusion: Generative Models in Dart

Diffusion is a Dart package building on top of `tensor` and `kamma` to provide state-of-the-art diffusion models like Stable Diffusion, directly in Dart.

## Features

-   **Diffusion Building Blocks**: Components like `ResnetBlock2D` for building custom diffusion networks.
-   **Stable Diffusion**: Support for loading and running Stable Diffusion pipelines (v1.5, v2.1).
-   **Device Agnostic**: Run on CPU, CUDA, or MPS.

## Usage

### Using Basic Blocks (Resnet2D)

```dart
import 'package:diffusion/diffusion.dart';
import 'package:tensor/tensor.dart';

void main() {
  final context = Context.best();

  // Initialize a random generator
  final generator = Generator.getDefault(device: context.device);
  generator.currentSeed = 0;

  // Create sample input and time embedding
  final sample = Tensor.randn([1, 32, 64, 64], device: context.device);
  final temb = Tensor.randn([1, 128], device: context.device);

  // Initialize a ResNet Block
  final resnet = ResnetBlock2D.make(
    numInChannels: 32,
    numOutChannels: 32,
    numTembChannels: 128,
  );
  
  // Forward pass
  final out = resnet.forward(sample, embeds: temb, context: context);
  print('Output Tensor:');
  print(out);
}
```

### Stable Diffusion Tokenization

```dart
import 'package:diffusion/diffusion.dart';
import 'package:kamma/kamma.dart';

Future<void> main() async {
  final prompt = 'minimalistic symmetrical logo with moose head';
  
  // Load CLIP Tokenizer
  final clip = await CLIPTokenizer.loadFromFile(
    'models/diffusion/bpe_simple_vocab_16e6.txt',
    config: ClipTextConfig.v2_1,
  );
  
  final tokens = clip.encode(prompt);
  print('Encoded Tokens: $tokens');
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
