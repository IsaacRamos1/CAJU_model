import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from earlystop import EarlyStopping
import matplotlib.pyplot as plt
import timm
import torch.nn.init as init


def initialize_weights_xavier(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

class LightSelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(LightSelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.feature_dim = feature_dim

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_probs = self.softmax(attention_scores)
        attended = torch.matmul(attention_probs, V)
        return attended

class CombinedVGG16MobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super(CombinedVGG16MobileNetV3, self).__init__()

        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg_features = vgg16.features

        for module in self.vgg_features.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.vgg_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.vgg_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.vgg_fc = nn.Linear(512, 512)

        mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet_features = mobilenet_v3_large.features
        self.mobilenet_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.mobilenet_maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.mobilenet_fc = nn.Linear(960, 512)

        self.attn_block = LightSelfAttention(1472)

        self.mlp = nn.Sequential(
            nn.Linear(1472, 2048),  
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x, return_embedding=False):

        x1 = self.vgg_features(x)        # [B, 512, H, W]
        x2 = self.mobilenet_features(x)  # [B, 960, H2, W2]

        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # Join
        x_combined = torch.cat([x1, x2], dim=1)  # [B, 1472, H, W]

        # Attention block
        B1, C1, H1, W1 = x_combined.shape
        x_combined_flat = x_combined.view(B1, C1, -1).transpose(1, 2)  # [B, H, W, C]
        x_combined_att = self.attn_block(x_combined_flat)
        x_combined_att = x_combined_att.transpose(1, 2).view(B1, C1, H1, W1)

        x_pooled = self.vgg_avgpool(x_combined_att)  # [B, 1472, 1, 1] Pooling global and flatten
        x_pooled = torch.flatten(x_pooled, 1)  # [B, 1472]
        
        x_combined = nn.functional.normalize(x_pooled, dim=1)
        if return_embedding:
            return x_combined

        output = self.mlp(x_combined)
        return output


class CNNTrainer:
    def __init__(self, model_name, fold, num_classes=4, device='cuda', patience=5):
        self.device = device   # gpu
        self.model = self._load_pretrained_model(model_name, num_classes)
        self.model.to(self.device)
        self.history = {'train_loss': [], 'val_accuracy': [], 'val_loss': []}
        self.early_stopping = EarlyStopping(patience=patience, fold=fold, verbose=True)
        self.models = ['vgg16', 'mobilenetv3', 'densenet161', 'inceptionv3', 'densenet121', 
               'MobileNetV3WithAttention', 'mobilenetv4_hybrid',
               'resnet101', 'efficientnet_b0', 'efficientnet_b3','resnet50'] # avaiable

        
        if model_name in self.models:
            for layer in self.convolutional_layers:
                for param in layer.parameters():
                    if not isinstance(layer, nn.BatchNorm2d):
                        param.requires_grad = False

        elif model_name == 'vgg16_mobilenetv3':
            for layer in self.model.vgg_features:
                for param in layer.parameters():
                    if not isinstance(layer, nn.BatchNorm2d):
                        param.requires_grad = False  # Freezing all convolutional layers for VGG16

            for layer in self.model.mobilenet_features:
                for param in layer.parameters():
                    if not isinstance(layer, nn.BatchNorm2d):
                        param.requires_grad = False  # Same for MobileNet
        

    def _load_pretrained_model(self, model_name, num_classes):
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            self.print_model_info(model)
        
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'mobilenetv3':
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            self.convolutional_layers = model.features  
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'vgg16_mobilenetv3':
            self.model = CombinedVGG16MobileNetV3(num_classes=num_classes)
            model = self.model
            self.print_model_info(model)

        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.convolutional_layers = model.features 
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'densenet161':
            model = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
            self.convolutional_layers = model.features  
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'inceptionv3':
            model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            self.convolutional_layers = nn.Sequential(*list(model.children())[:-1])
            self.print_model_info(model)
        
        elif model_name == 'densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.convolutional_layers = model.features
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            self.print_model_info(model)
        
        elif model_name == 'mobilenetv4_hybrid':
            model = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True)
            in_features = model.get_classifier().in_features
            model.reset_classifier(num_classes)
            self.convolutional_layers = list(model.children())[:-1]  # ou parte inicial da rede, se desejar congelar
            self.print_model_info(model)
        
        elif model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.convolutional_layers = nn.Sequential(*list(model.children())[:-2])  # Para Grad-CAM e descongelamento
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.convolutional_layers = model.features
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            self.print_model_info(model)

        elif model_name == 'efficientnet_b3':
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            self.convolutional_layers = model.features
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            self.print_model_info(model)
        else:
            raise ValueError(f"Model '{model_name}' not found.")
        
        model.apply(initialize_weights_xavier)
        return model
    
    def print_model_info(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)

        print("Active device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        print(f"Num parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_mb:.2f} MB")

    def train(self, trainloader, valloader, epochs=10, lr=0.001, fold=0,
            log_callback=None, plot_callback=None, stop_flag_getter=lambda: False,
            project_name='project-name', batch_size=None):

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

        torch.cuda.empty_cache()

        def safe_log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        for epoch in range(epochs):
            if stop_flag_getter and stop_flag_getter():
                safe_log("Training process interrupted by user.")
                break

            if epoch == 5:
                print("Unfreezing convolutional layers...")
                if hasattr(self, 'convolutional_layers'):
                    for layer in self.convolutional_layers:
                        for param in layer.parameters():
                            param.requires_grad = True
                elif hasattr(self.model, 'vgg_features') and hasattr(self.model, 'mobilenet_features'):
                    for layer in self.model.vgg_features:
                        for param in layer.parameters():
                            param.requires_grad = True
                    for layer in self.model.mobilenet_features:
                        for param in layer.parameters():
                            param.requires_grad = True
                del optimizer
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

            self.model.train()
            running_loss = 0.0
            safe_log(f"\nFold {fold} | Epoch {epoch+1}/{epochs}")
            safe_log("--------------------------------------------------------------------------------")

            for inputs, labels in tqdm(trainloader, desc=f"Train Epoch {epoch+1}/{epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                if isinstance(outputs, tuple):
                    main_output, aux_output = outputs
                    loss = criterion(main_output, labels) + 0.4 * criterion(aux_output, labels)
                    outputs = main_output
                else:
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(trainloader)

            val_loss, val_acc, val_prec, val_recall, val_f1, val_auc = self.evaluate(valloader)
            safe_log(f"Train Loss: {avg_train_loss:.4f}  |  Val acc: {val_acc:4f}")

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                safe_log("Early stopping!")
                break

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_loss'].append(val_loss)

            if log_callback:
                log_callback(f"[DEBUG] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

            if plot_callback:
                plot_callback(self.history['train_loss'], self.history['val_accuracy'])

        self.plot_training_metrics(fold=fold, save_path=f"metrics_fold{fold}.png")

    

    def evaluate(self, dataloader, type='Validacao'):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_outputs = []
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=type):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.softmax(dim=1).cpu().numpy())

        accuracy =  correct / total
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        avg_loss = running_loss / len(dataloader)

        try:
            auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr', average='macro')
        except ValueError:
            auc = float('nan')

        return avg_loss, accuracy, precision, recall, f1, auc

    def predict(self, inputs):
        print("Iniciando Teste...")
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu()
    
    def plot_training_metrics(self, fold=0, save_path='resultados/metrics_plot.png'):
        epochs = range(1, len(self.history['train_loss']) + 1)
        train_loss = self.history['train_loss']
        val_loss = self.history['val_loss']
        val_acc = self.history['val_accuracy']
        
        # Encontra a época com menor validação loss
        min_loss_epoch = val_loss.index(min(val_loss)) + 1

        plt.figure(figsize=(12, 6))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Loss de Treinamento')
        plt.plot(epochs, val_loss, label='Loss de Validação')
        plt.axvline(min_loss_epoch, color='gray', linestyle='--', label=f'Mínimo Loss (Época {min_loss_epoch})')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Loss por Época')
        plt.legend()
        plt.grid(True)

        # Val
        plt.subplot(1, 2, 2)
        plt.plot(epochs, val_acc, label='Acurácia de Validação', color='green')
        plt.axvline(min_loss_epoch, color='gray', linestyle='--', label=f'Mínimo Loss (Época {min_loss_epoch})')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.title('Acurácia por Época')
        plt.legend()
        plt.grid(True)

        plt.suptitle(f'Métricas de Treinamento - Fold {fold}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path)
        plt.close()

        print(f"📈 Gráfico salvo em: {save_path}")
