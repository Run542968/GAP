


def build(args, device):
    if args.target_type != "none": # adopt one-hot as target, only used in close_set
        num_classes = int(args.num_classes * args.split / 100)
    else:
        num_classes = args.num_classes

    if args.feature_type == "ViFi-CLIP":
        text_encoder,logit_scale = None, torch.from_numpy(np.load(os.path.join(args.feature_path,'logit_scale.npy'))).float()
    elif args.feature_type == "CLIP":
        text_encoder, logit_scale = build_text_encoder(args,device)
    else:
        raise NotImplementedError
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    # semantic_vhead,semantic_thead = build_semantic_head(args)
    if args.enable_refine:
        refine_decoder = build_refine_decoder(args)
    else:
        refine_decoder = None

    model = ConditionalDETR(
        backbone,
        transformer,
        text_encoder,
        refine_decoder,
        logit_scale,
        device=device,
        num_classes=num_classes,
        args=args
    )
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.actionness_loss or args.eval_proposal or args.enable_classAgnostic:
        weight_dict['loss_actionness'] = args.actionness_loss_coef
    if args.salient_loss:
        weight_dict['loss_salient'] = args.salient_loss_coef
    if args.adapterCLS_loss:
        weight_dict['loss_adapterCLS'] = args.adapterCLS_loss_coef
    if args.refine_actionness_loss:
        weight_dict['loss_actionness_refine'] = args.refine_actionness_loss_coef
        weight_dict['loss_bbox_refine'] = args.refine_actionness_loss_coef
    if args.distillation_loss:
        weight_dict['loss_distillation'] = args.distillation_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    criterion = build_criterion(args, num_classes, matcher=matcher, weight_dict=weight_dict)
    criterion.to(device)

    postprocessor = build_postprocess(args)

    return model, criterion, postprocessor
