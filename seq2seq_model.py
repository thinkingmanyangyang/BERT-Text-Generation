import random
from modeling_unilm import *
from transformers import BertPreTrainedModel
from loss_function import LabelSmoothLoss
from utils import BeamHypotheses


class Seq2SeqModel(BertPreTrainedModel):
    def __init__(self, config, tokenizer, output_max_length=60):
        super(Seq2SeqModel, self).__init__(config)
        self.tokenizer = tokenizer
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        # self.loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        self.loss_fct = LabelSmoothLoss(smoothing=0.1)
        self.output_max_length = output_max_length

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                answer_tag_ids=None):
        outputs = self.forward_step(
            # input_ids=input_ids,
            input_ids=self.random_choice(input_ids, token_type_ids, attention_mask),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            answer_tag_ids=answer_tag_ids,
        )
        predictions = outputs[:, :-1, :].contiguous()
        output_ids = input_ids[:, 1:].contiguous()
        output_mask = token_type_ids[:, 1:].contiguous()
        # loss = self.compute_loss_for_sentence(predictions, labels=output_ids, target_mask=output_mask)
        loss = self.compute_loss(predictions, labels=output_ids, target_mask=output_mask)
        return loss

    def forward_step(self,
                     input_ids,
                     token_type_ids,
                     position_ids,
                     attention_mask,
                     answer_tag_ids=None):
        # the step for one time forward
        extend_attention_mask = self.create_attention_mask(
            token_type_ids,
            attention_mask,
        )
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=extend_attention_mask,
            answer_tag_ids=answer_tag_ids,
        )
        sequence_output = output[0]
        predictions = self.cls(sequence_output)
        return predictions

    def random_choice(self, input_ids, token_type_ids, attention_mask, type=1):
        # type 0, no random choice
        # type 1, target sentence random choice
        # type 2, source and target all use random choice, in this type,
        #         you should use mlm task for source sentence
        if type == 0:
            return input_ids
        # random choice
        if random.random() < 0.5:
            input_ids = self._random_choice(input_ids, target_mask=token_type_ids)
        elif type == 2:
            input_ids = self._random_choice(
                input_ids=input_ids,
                target_mask=(1 - token_type_ids) & attention_mask
            )
        return input_ids

    def _random_choice(self, input_ids, target_mask):
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device
        vocab_size = self.bert.embeddings.word_embeddings.weight.shape[0]
        rand_ids = torch.rand(batch_size, sequence_length) * vocab_size
        rand_ids = rand_ids.to(input_ids.dtype).to(device) % vocab_size
        rand_mask = torch.rand(batch_size, sequence_length).to(device) > 0.3
        disturb_ids = torch.where(rand_mask, input_ids, rand_ids)
        input_ids = torch.where(target_mask == 0, input_ids, disturb_ids)
        return input_ids

    def compute_loss(self, predictions, labels, target_mask):
        # compute loss for batch
        vocab_size = predictions.shape[-1]
        predictions = predictions.view(-1, vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = self.loss_fct(predictions, labels) * target_mask
        return loss.sum() / target_mask.sum()

    def compute_loss_for_sentence(self, predictions, labels, target_mask):
        # compute loss for sentence, then compute the mean of loss
        batch_size = predictions.shape[0]
        loss = 0.0
        for i in range(batch_size):
            loss += self.compute_loss(predictions[i], labels[i], target_mask[i])
        return loss / batch_size

    def create_attention_mask(self, token_type_ids, attention_mask):
        idxs = torch.cumsum(token_type_ids, dim=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = mask.to(token_type_ids.dtype)
        attention_mask = attention_mask[:, None, :]
        return mask & attention_mask

    def beam_serach_predict(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                beam_size=5):
        # too slow...
        # predict sentence one by one.
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device
        output_ids = torch.empty(beam_size, 0, dtype=input_ids.dtype, device=device)
        output_scores = torch.zeros(beam_size, dtype=torch.float, device=device)
        output_scores[1:] = -1e9
        generated_hyps = BeamHypotheses(num_beams=beam_size, max_length=self.output_max_length, length_penalty=1)
        vocab_size = self.bert.embeddings.word_embeddings.weight.shape[0]
        end_id = self.tokenizer.token_to_id(self.tokenizer._token_end)
        # end_id = self.tokenizer.token_to_id('？')

        for step in range(self.output_max_length):
            if step == 0:
                input_ids = input_ids.repeat(beam_size, 1)
                token_type_ids = token_type_ids.repeat(beam_size, 1)
                position_ids = position_ids.repeat(beam_size, 1)
                attention_mask = attention_mask.repeat(beam_size, 1)
            predictions = self.forward_step(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            pred_scores = torch.log_softmax(predictions[:, -1, :], dim=-1)
            next_scores = output_scores[:, None] + pred_scores
            next_scores = next_scores.reshape(-1)

            next_scores, next_ids = torch.topk(next_scores, beam_size, dim=-1, largest=True, sorted=True)
            beam_ids = next_ids // vocab_size
            token_ids = next_ids % vocab_size
            for beam_id, token_id, next_score in zip(beam_ids, token_ids, next_scores):
                if token_id == end_id:
                    generated_hyps.add(output_ids[beam_id].tolist(), next_score.item())

            if generated_hyps.is_done(next_scores[0], step):
                break
            pred_ids = token_ids.reshape(beam_size, -1)
            # reorder input and output
            output_ids = output_ids[beam_ids, :]
            output_scores = next_scores.view(-1)

            input_ids = input_ids[beam_ids, :]
            token_type_ids = token_type_ids[beam_ids, :]
            position_ids = position_ids[beam_ids, :]
            attention_mask = attention_mask[beam_ids, :]
            # add new id in output ids
            output_ids = torch.cat([output_ids, pred_ids], dim=-1)

            input_ids = torch.cat([input_ids, pred_ids], dim=-1)
            token_type_ids = torch.cat([token_type_ids, torch.ones_like(pred_ids)], dim=-1)
            position_ids = torch.cat([position_ids, position_ids[:, -1:]+1], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(pred_ids)], dim=-1)

        sorted_hyps = sorted(generated_hyps.beams, key=lambda x: x[0])
        try:
            result = sorted_hyps.pop()[1]
        except:
            result = output_ids[0].tolist()
        return [result]

    def predict(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                answer_tag_ids=None,
                ):
        # predict sentence one by one.
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device
        output_ids = torch.empty(batch_size, 0, dtype=input_ids.dtype, device=device)
        end_id = self.tokenizer.token_to_id(self.tokenizer._token_end)
        # end_id = self.tokenizer.token_to_id('？')

        for step in range(self.output_max_length):
            predictions = self.forward_step(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                answer_tag_ids=answer_tag_ids
            )
            _, pred_ids = torch.max(predictions[:, -1, :], dim=-1, keepdim=True)
            if pred_ids == end_id:
                break

            output_ids = torch.cat([output_ids, pred_ids], dim=-1)

            input_ids = torch.cat([input_ids, pred_ids], dim=-1)
            token_type_ids = torch.cat([token_type_ids, torch.ones_like(pred_ids)], dim=-1)
            position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(pred_ids)], dim=-1)
            if answer_tag_ids is not None:
                answer_tag_ids = torch.cat([answer_tag_ids, torch.ones_like(pred_ids)], dim=-1)
        return output_ids.tolist()









