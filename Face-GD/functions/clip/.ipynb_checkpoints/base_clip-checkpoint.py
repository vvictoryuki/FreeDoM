import torch
import torch.nn as nn
from .clip import clip
# from clip import clip
import torchvision
# from mmedit.models.registry import COMPONENTS

model_name = "ViT-B/16"
# model_name = "ViT-B/32"
# ref_image_feature = torch.load("/userhome/yjw/mmediting/fengge/02583.pth")


def load_clip_to_cpu():
    url = clip._MODELS[model_name]
    model_path = clip._download(url)
#     print("------------------>", model_path)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class CustomCLIP(nn.Module):
    def __init__(self, prompt, device, need_tokenize=True):
        super().__init__()
        clip_model = load_clip_to_cpu().to(device)
        for name, param in clip_model.named_parameters():
            param.requires_grad_(False)

        if need_tokenize:
            text = clip.tokenize(prompt).to(device)
            self.text_features_con = clip_model.encode_text(text)
        else:
            text_list = ["X X X.", "Y Y Y."]
            text = clip.tokenize(text_list).to(device)
            text_embedding_dict = prompt
            ctx = text_embedding_dict['ctx']
            token_prefix = text_embedding_dict['token_prefix']
            token_suffix = text_embedding_dict['token_suffix']
            text_embedding = torch.cat([token_prefix, ctx, token_suffix], dim=1).to(device)
            self.text_features_con = clip_model.encode_text_embedding(text_embedding, text)

        self.text_features_con = clip_model.encode_text(text)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features_con / self.text_features_con.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, opt):
        super().__init__()
        n_ctx = opt['n_words']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # init from given sentences
        if opt['use_init']:

            prompt_pos = clip.tokenize(opt['ctx_init'][1])
            with torch.no_grad():
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx, :]

            prompt_neg = clip.tokenize(opt['ctx_init'][0])
            with torch.no_grad():
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx, :]

            prompt_prefix_pos = opt['ctx_init'][1]
            prompt_prefix_neg = opt['ctx_init'][0]

        # random initialization
        else:
            print("Initializing a generic context")
            ctx_vectors_pos = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            ctx_vectors_neg = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx)
            prompt_prefix_neg = " ".join(["Y"] * n_ctx)

        print(f'Pos initial context: "{prompt_prefix_pos}"')
        print(f'Neg initial context: "{prompt_prefix_neg}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # self.ctx = nn.Parameter(ctx_vectors_pos)  # to be optimized
        # self.ctx = nn.Parameter(ctx_vectors_neg)  # to be optimized
        self.ctx = nn.Parameter(
            torch.cat(
                [
                    ctx_vectors_neg.unsqueeze(0),
                    ctx_vectors_pos.unsqueeze(0),
                ],
                dim=0
            )
        )

        prompts = [prompt_prefix_pos + ".", prompt_prefix_neg + "."]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        # print("--->", ctx.size(), prefix.size(), suffix.size())

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CLIPGAN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        clip_model = load_clip_to_cpu()
        self.prompt_learner = PromptLearner(clip_model, opt)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        )

    def forward(self, image):
        image = self.preprocess(image)
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CLIPEncoder(nn.Module):
    def __init__(self, need_ref=False, ref_path=None):
        super().__init__()
        self.clip_model = load_clip_to_cpu()
        self.clip_model.requires_grad = True
        # self.preprocess = torchvision.transforms.Normalize(
        #     (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        # )
        self.preprocess = torchvision.transforms.Normalize(
            (0.48145466*2-1, 0.4578275*2-1, 0.40821073*2-1),
            (0.26862954*2, 0.26130258*2, 0.27577711*2)
        )
        if need_ref:
            self.to_tensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            
            ref_path = "/userhome/yjw/ddgm_exp/functions/clip/xiangrikui.jpg" if not ref_path else ref_path
#             content_path = "/userhome/yjw/ddgm_exp/functions/clip/apple.png"
            
            from PIL import Image
            
            img = Image.open(ref_path).convert('RGB')
            image = img.resize((224, 224), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            self.ref = img
            
#             img = Image.open(content_path).convert('RGB')
#             image = img.resize((256, 256), Image.BICUBIC)
#             img = self.to_tensor(image)
#             img = torch.unsqueeze(img, 0)
#             img = img.cuda()
#             self.content = img
    
    def get_residual(self, image, text):
        text = clip.tokenize(text).cuda()
        image = torch.nn.functional.interpolate(image, size=224, mode='bicubic')
        image = self.preprocess(image)
        image_feature, _ = self.clip_model.encode_image_with_features(image)
        text_feature = self.clip_model.encode_text(text)
        text_feature = text_feature.repeat(image.shape[0], 1)
        return text_feature - image_feature
#         image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
#         text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
#         return 1. - image_feature @ text_feature.t()
    
    def get_gram_matrix_residual(self, im1):
        im1 = torch.nn.functional.interpolate(im1, size=224, mode='bicubic')
        im1 = self.preprocess(im1)
        _, feats1 = self.clip_model.encode_image_with_features(im1)
        
#         for i, feat in enumerate(feats1):
#             print(i, feat[1:, :, :].size())
        
        _, feats2 = self.clip_model.encode_image_with_features(self.ref)
        
        feat1 = feats1[2][1:, 0, :]
        feat2 = feats2[2][1:, 0, :]
        
        gram1 = torch.mm(feat1.t(), feat1)
        gram2 = torch.mm(feat2.t(), feat2)
        
#         print(gram1.size(), gram2.size())
        return gram1 - gram2

    def get_content_residual(self, im1):
#         im1 = torch.nn.functional.interpolate(im1, size=224, mode='bicubic')
        im1 = self.preprocess(im1)
        return im1 - self.content
    
    def forward(self, image, text=None):
        if text is not None:
            text = clip.tokenize(text).cuda()
            return self.encode_image_with_text(image, text)
        else:
            return self.encode_image_with_features(image)


if __name__ == "__main__":
    m = CLIPEncoder().cuda()
    im1 = torch.randn((1, 3, 224, 224)).cuda()
    im2 = torch.randn((1, 3, 224, 224)).cuda()
    m.get_gram_matrix_residual(im1, im2)
