#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_set>
#include <unordered_map>

#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/dict.h"
#include "dynet/cfsm-builder.h"
#include "dynet/io.h"

#include "impl/oracle.h"
#include "impl/cl-args.h"


int parseLine(char* line){
    int i = strlen(line);
    while (*line < '0' || *line > '9') line++;
    line[i-3] = '\0';
    i = atoi(line);
    return i;
}

int getMemoryUsage(){ //Note: this value is in KB!
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];


    while (fgets(line, 128, file) != NULL){
        if (strncmp(line, "VmSize:", 7) == 0){
            result = parseLine(line);
            break;
        }
    }
    fclose(file);
    return result;
}


// dictionaries
dynet::Dict termdict, ntermdict, adict, posdict;

volatile bool requested_stop =false;
unsigned ACTION_SIZE = 0;
unsigned VOCAB_SIZE = 0;
unsigned NT_SIZE = 0;
unsigned POS_SIZE = 0;

bool DYNAMIC_BEAM = false;
int beam=5;

//bool MULTITHREAD_BEAMS = false;
//boost::mutex cnn_mutex;


std::map<int,int> action2NTindex;  // pass in index of action NT(X), return index of X
using namespace dynet;
using namespace std;
Params params;
unordered_map<unsigned, vector<float>> pretrained;
vector<bool> singletons; // used during training

vector<unsigned> possible_actions;

struct ParserBuilder {

  LSTMBuilder state_lstm;
  LSTMBuilder l2rbuilder;
  LSTMBuilder r2lbuilder;
  LookupParameter p_w; // word embeddings
  LookupParameter p_t; // pretrained word embeddings (not updated)
  LookupParameter p_a; // input action embeddings
  LookupParameter p_r; // relation embeddings
  LookupParameter p_p; // pos tag embeddings
  Parameter p_w2l; // word to LSTM input
  Parameter p_p2l; // POS to LSTM input
  Parameter p_t2l; // pretrained word embeddings to LSTM input
  Parameter p_lb; // LSTM input bias

  Parameter p_sent_start;
  Parameter p_sent_end;

  Parameter p_s_input2att;
  Parameter p_s_h2att;
  Parameter p_s_attbias;
  Parameter p_s_att2attexp;
  Parameter p_s_att2combo;
  
  Parameter p_b_input2att;
  Parameter p_b_h2att;
  Parameter p_b_attbias;
  Parameter p_b_att2attexp;
  Parameter p_b_att2combo;

  Parameter p_h2combo;
  Parameter p_combobias;
  Parameter p_combo2rt;
  Parameter p_rtbias;

  explicit ParserBuilder(Model* model, const unordered_map<unsigned, vector<float>>& pretrained) :
      state_lstm(1, params.state_input_dim ,params.state_hidden_dim, *model),
      l2rbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      r2lbuilder(params.layers, params.bilstm_input_dim, params.bilstm_hidden_dim, *model),
      p_w(model->add_lookup_parameters(VOCAB_SIZE, {params.input_dim})),
      p_a(model->add_lookup_parameters(ACTION_SIZE, {params.action_dim})),
      p_r(model->add_lookup_parameters(ACTION_SIZE, {params.rel_dim})),
      p_w2l(model->add_parameters({params.bilstm_input_dim, params.input_dim})),
      p_lb(model->add_parameters({params.bilstm_input_dim})),
      p_sent_start(model->add_parameters({params.bilstm_input_dim})),
      p_sent_end(model->add_parameters({params.bilstm_input_dim})),
      p_s_input2att(model->add_parameters({params.attention_hidden_dim, params.bilstm_hidden_dim*2})),
      p_s_h2att(model->add_parameters({params.attention_hidden_dim, params.state_hidden_dim})),
      p_s_attbias(model->add_parameters({params.attention_hidden_dim})),
      p_s_att2attexp(model->add_parameters({params.attention_hidden_dim})),
      p_s_att2combo(model->add_parameters({params.state_hidden_dim, params.bilstm_hidden_dim*2})),
      p_b_input2att(model->add_parameters({params.attention_hidden_dim, params.bilstm_hidden_dim*2})),
      p_b_h2att(model->add_parameters({params.attention_hidden_dim, params.state_hidden_dim})),
      p_b_attbias(model->add_parameters({params.attention_hidden_dim})),
      p_b_att2attexp(model->add_parameters({params.attention_hidden_dim})),
      p_b_att2combo(model->add_parameters({params.state_hidden_dim, params.bilstm_hidden_dim*2})),
      p_h2combo(model->add_parameters({params.state_hidden_dim, params.state_hidden_dim})),
      p_combobias(model->add_parameters({params.state_hidden_dim})),
      p_combo2rt(model->add_parameters({ACTION_SIZE, params.state_hidden_dim})),
      p_rtbias(model->add_parameters({ACTION_SIZE})){

    if (params.use_pos) {
      p_p = model->add_lookup_parameters(POS_SIZE, {params.pos_dim});
      p_p2l = model->add_parameters({params.bilstm_input_dim, params.pos_dim});
    }
//    buffer_lstm = new LSTMBuilder(params.layers, LSTM_params.input_dim, HIDDEN_DIM, model);
    if (pretrained.size() > 0) {
      p_t = model->add_lookup_parameters(VOCAB_SIZE, {params.pretrained_dim});
      for (auto it : pretrained)
        p_t.initialize(it.first, it.second);
      p_t2l = model->add_parameters({params.bilstm_input_dim, params.pretrained_dim});
    }
  }




  struct Action {
      unsigned val;
      double score;
      Expression log_prob;

      //Expression log_zlocal;
      //Expression rho;
  };

  struct ActionCompare {
      bool operator()(const Action& a, const Action& b) const {
          return a.score > b.score;
      }
  };


  struct ParserState {
    vector<int> stacki;
    vector<int> bufferi;
//    LSTMBuilder l2rbuilder;
//    LSTMBuilder r2lbuilder;
    LSTMBuilder state_lstm;

    vector<unsigned> results;  // sequence of predicted actions
    vector<Expression> log_probs;
    double score;
    int nopen_parens;
    unsigned nt_count;
    vector<int> is_open_paren;
    char prev_a;
    unsigned stack_buffer_split;

    //For beam-search
    bool complete;

    bool gold = true;
    Action next_gold_action;

    Expression s_att_pool;
    Expression b_att_pool;
    unsigned unary;

  };


  struct ParserStateCompare {
    bool operator()(const ParserState& a, const ParserState& b) const {
      return a.score > b.score;
    }
  };


  struct ParserStatePointerCompare {
      bool operator()(ParserState* a, ParserState* b) const {
          return a->score > b->score;
      }
  };

  struct ParserStatePointerCompareReverse {
      bool operator()(ParserState* a, ParserState* b) const {
          return a->score < b->score;
      }
  };


  struct StepSelect {
      Action action;
      double total_score;
      ParserState* source;
  };

  struct StepSelectCompare {
      bool operator()(const StepSelect& a, const StepSelect& b) const {
          return a.total_score > b.total_score;
      }
  };


  void apply_action_to_state(  ComputationGraph* hg,
                                 ParserState* ns,
								 //const Expression& s_att_pool,
								 //const Expression& b_att_pool,
                                 unsigned action) {

//        apply_action(hg, ns->state_lstm, s_att_pool, b_att_pool,
//                     ns->bufferi, ns->stacki, ns->results,
//                     action, ns->is_open_paren, ns->stack_buffer_split, ns->nopen_parens, ns->nt_count, ns->prev_a);

	  apply_action(hg, ns->state_lstm, ns->s_att_pool, ns->b_att_pool,
	                       ns->bufferi, ns->stacki, ns->results,
	                       action, ns->is_open_paren, ns->stack_buffer_split, ns->nopen_parens, ns->nt_count, ns->prev_a, ns->unary);

    }



    void apply_action( ComputationGraph* hg,
    		   LSTMBuilder& state_lstm,
    		   const Expression& s_att_pool,
    		   const Expression& b_att_pool,
                       vector<int>& bufferi,
                       vector<int>& stacki,
                       vector<unsigned>& results,
                       unsigned action,
					   vector<int>& is_open_paren,
    				   unsigned& stack_buffer_split,
    				   int& nopen_parens,
    				   unsigned& nt_count,
    				   char& prev_a,
					   unsigned& unary) {


   

        	 results.push_back(action);


          // add current action to action LSTM
          Expression actione = lookup(*hg, p_a, action);
          state_lstm.add_input(concatenate({actione,s_att_pool,b_att_pool}));

          // do action
          const string& actionString=adict.convert(action);
          //cerr << "ACT: " << actionString <<" " <<stack_buffer_split << endl;
          const char ac = actionString[0];
          const char ac2 = actionString[1];
          if (ac =='S' && ac2=='H') {  // SHIFT
            assert(bufferi.size() > 1); // dummy symbol means > 1 (not >= 1)
            stacki.push_back(bufferi.back());
            bufferi.pop_back();
            is_open_paren.push_back(-1);
            stack_buffer_split += 1;
            //ADDED
            unary = 0;

          } else if (ac == 'N') { // NT
            ++nopen_parens;
            assert(stacki.size() > 1);
            auto it = action2NTindex.find(action);
            assert(it != action2NTindex.end());
            int nt_index = it->second;
            nt_count++;
            stacki.push_back(-1);
            is_open_paren.push_back(nt_index);
          } else  if (ac == 'R'){ // REDUCE
            --nopen_parens;

				//ADDED
				if(prev_a == 'N') unary += 1;
				if(prev_a == 'R') unary = 0;

					assert(stacki.size() > 2); // dummy symbol means > 2 (not >= 2)
					// find what paren we are closing
					int i = is_open_paren.size() - 1;
					while(is_open_paren[i] < 0) { --i; assert(i >= 0); }
					int nchildren = is_open_paren.size() - i - 1;
					//assert(nchildren > 0);
				assert(nchildren+1 > 0);//ADDED
					//cerr << "  number of children to reduce: " << nchildren << endl;

					// REMOVE EVERYTHING FROM THE STACK THAT IS GOING
					// TO BE COMPOSED INTO A TREE EMBEDDING
					for (i = 0; i < nchildren; ++i) {
					  assert (stacki.back() != -1);
					  stacki.pop_back();
					  is_open_paren.pop_back();
					}

				is_open_paren.pop_back(); // nt symbol
					stacki.pop_back(); // nonterminal dummy

				//ADDED
				stacki.pop_back();//leftmost
				is_open_paren.pop_back();

					// BUILD TREE EMBEDDING USING BIDIR LSTM
					stacki.push_back(999); // who knows, should get rid of this
					is_open_paren.push_back(-1); // we just closed a paren at this position
          }else{//TERM
    	}

    	prev_a = ac;


    //      ns->stack_buffer_split=stack_buffer_split;
    //      ns->nopen_parens=nopen_parens;
    //      ns->nt_count=nt_count;
    //      ns->prev_a=prev_a;

    }



    static bool IsActionForbidden_Discriminative(const string& a, char prev_a, unsigned bsize, unsigned ssize, unsigned nopen_parens, unsigned unary) {
      bool is_shift = (a[0] == 'S' && a[1]=='H');
      bool is_reduce = (a[0] == 'R' && a[1]=='E');
      bool is_nt = (a[0] == 'N');
      bool is_term = (a[0] == 'T');
      //cerr << "AAAAAA " << a<<"\n";
      assert(is_shift || is_reduce || is_nt || is_term) ;
      static const unsigned MAX_OPEN_NTS = 100;
      static const unsigned MAX_UNARY = 3;
    //  if (is_nt && nopen_parens > MAX_OPEN_NTS) return true;
      if (is_term){
        if(ssize == 2 && bsize == 1 && prev_a == 'R') return false;
        return true;
      }

      if(ssize == 1){
         if(!is_shift) return true;
         return false;
      }

      if (is_shift){
        if(bsize == 1) return true;
        if(nopen_parens == 0) return true;
        return false;
      }

      if (is_nt) {
        if(bsize == 1 && unary >= MAX_UNARY) return true;
        if(prev_a == 'N') return true;
        return false;
      }

      if (is_reduce){
        if(unary > MAX_UNARY) return true;
        if(nopen_parens == 0) return true;
        return false;
      }

      // TODO should we control the depth of the parse in some way? i.e., as long as there
      // are items in the buffer, we can do an NT operation, which could cause trouble
    }



    struct getNextBeamsArgs {
        const Expression& s_attbias;
        const Expression& s_input2att;
        const Expression& sent_start_expr;
        const Expression& s_h2att;
        const Expression& b_attbias;
        const Expression& b_input2att;
        const Expression& sent_end_expr;
        const Expression& b_h2att;
        const Expression& s_att2combo;
        const Expression& b_att2combo;
        const Expression& h2combo;
        const Expression& combobias;
        const Expression& combo2rt;
        const Expression& rtbias;
        const Expression& s_att2attexp;
        const Expression& b_att2attexp;
//        Expression& s_att_pool;
//        Expression& b_att_pool;
        const bool& train;
        const vector<int>& correct_actions;
        const int& action_count;
        const int& sent_size;
        vector<Expression> input;
    };



    static void getNextBeams(ParserState* cur, vector<StepSelect>* potential_next_beams,
                                    ComputationGraph* hg,
    				    const getNextBeamsArgs& args,
                                    ParserState*& gold_parse){


        const Expression& s_attbias=args.s_attbias;
        const Expression& s_input2att=args.s_input2att;
        const Expression& sent_start_expr=args.sent_start_expr;
        const Expression& s_h2att=args.s_h2att;
        const Expression& b_attbias=args.b_attbias;
        const Expression& b_input2att=args.b_input2att;
        const Expression& sent_end_expr=args.sent_end_expr;
        const Expression& b_h2att=args.b_h2att;
        const Expression& s_att2combo=args.s_att2combo;
        const Expression& b_att2combo=args.b_att2combo;
        const Expression& h2combo=args.h2combo;
        const Expression& combobias=args.combobias;
        const Expression& combo2rt=args.combo2rt;
        const Expression& rtbias=args.rtbias;
        const Expression& s_att2attexp=args.s_att2attexp;
        const Expression& b_att2attexp=args.b_att2attexp;
//        Expression& s_att_pool=args.s_att_pool;
//        Expression& b_att_pool=args.b_att_pool;
        const bool& train=args.train;
        const vector<int>& correct_actions=args.correct_actions;
        const int& action_count=args.action_count;
        const int& sent_size=args.sent_size;
        const vector<Expression>& input=args.input;



        if(params.debug)cerr<<"ENTRA GET BEAMS"<<endl;

          vector<unsigned> current_valid_actions;
          for (auto a : possible_actions) {
            if (IsActionForbidden_Discriminative(adict.convert(a), cur->prev_a, cur->bufferi.size(), cur->stacki.size(), cur->nopen_parens, cur->unary))
              continue;

            if(params.debug)cerr<<"VALID "<<adict.convert(a)<<endl;
              current_valid_actions.push_back(a);
          }


//          cnn_mutex.lock();

          Expression prev_h = cur->state_lstm.final_h()[0];
          vector<Expression> s_att;
          vector<Expression> s_input;
          s_att.push_back(tanh(affine_transform({s_attbias, s_input2att, sent_start_expr, s_h2att, prev_h})));
          s_input.push_back(sent_start_expr);
          for(unsigned i = 0; i < cur->stack_buffer_split; i ++){
            s_att.push_back(tanh(affine_transform({s_attbias, s_input2att, input[i], s_h2att, prev_h})));
            s_input.push_back(input[i]);
          }
          Expression s_att_col = transpose(concatenate_cols(s_att));
          Expression s_attexp = softmax(s_att_col * s_att2attexp);

          Expression s_input_col = concatenate_cols(s_input);
          Expression s_att_pool = s_input_col * s_attexp;




          vector<Expression> b_att;
          vector<Expression> b_input;
          for(unsigned i = cur->stack_buffer_split; i < sent_size; i ++){
            b_att.push_back(tanh(affine_transform({b_attbias, b_input2att, input[i], b_h2att, prev_h})));
            b_input.push_back(input[i]);
          }
          b_att.push_back(tanh(affine_transform({b_attbias, b_input2att, sent_end_expr, b_h2att, prev_h})));
          b_input.push_back(sent_end_expr);
          Expression b_att_col = transpose(concatenate_cols(b_att));
          Expression b_attexp = softmax(b_att_col * b_att2attexp);

          Expression b_input_col = concatenate_cols(b_input);
          Expression b_att_pool = b_input_col * b_attexp;




          Expression combo = affine_transform({combobias, h2combo, prev_h, s_att2combo, s_att_pool, b_att2combo, b_att_pool});
          Expression n_combo = rectify(combo);
          Expression r_t = affine_transform({rtbias, combo2rt, n_combo});



          //ADDED
          cur->s_att_pool=s_att_pool;
          cur->b_att_pool=b_att_pool;


          Expression r_t_s = select_rows(r_t, current_valid_actions);
          Expression adiste = log_softmax(r_t_s);


          vector<float> adist = as_vector(hg->incremental_forward(adiste));

//          double best_score = adist[0];
//          unsigned model_action = current_valid_actions[0];



    	for (unsigned i = 0; i < current_valid_actions.size(); ++i) {
    		// For each action, its value is equal to the current state's value, plus the value of the action
    		double total_score = cur->score + adist[i];


    		//if(params.debug)cerr<<i<<"  "<<	current_valid_actions[i]<<" "<<adict.convert(current_valid_actions[i])<<endl;
    		//cerr << "filling\n";
    		Action act;
    		act.score = adist[i];
    		act.val = current_valid_actions[i];
    		//act.log_prob = pick(adiste, act.val);
    		act.log_prob = pick(adiste, i);

    		StepSelect next_step;
    		next_step.source = cur;
    		next_step.action = act;
    		next_step.total_score = total_score;

    		// if it is gold, give the gold act
    		//Se incluye la gold tran
    		if (train && cur->gold) {
    		    Action gold_act;
    		    if(params.debug)cerr << "ACTION COUNT "<<action_count<< "\n";
    		    gold_act.val = correct_actions[action_count];
    		    unsigned w = 0;
    		    for(;w< current_valid_actions.size(); w++){
    		           if(current_valid_actions[w] == correct_actions[action_count]) break;
    		    }
    		    gold_act.score = adist[w];
    		    gold_act.log_prob = pick(adiste, w);
    		    gold_parse = cur;
    		    gold_parse->next_gold_action = gold_act;

    		    if(params.debug)cerr<<"NEXT GOLD ACT IS>>> "<<adict.convert(current_valid_actions[w])<<endl;

    		}
    		if(params.debug)cerr<<"ADDED AS NEXT "<<adict.convert(current_valid_actions[i])<<endl;

    		potential_next_beams->push_back(next_step);
       	 }

        current_valid_actions.clear();
    	


//       cnn_mutex.unlock();
    };






Expression log_prob_parser(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
		     vector<unsigned>& results,
		     bool train,
                     bool sample = false) {
    


    l2rbuilder.new_graph(*hg);
    r2lbuilder.new_graph(*hg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression lb = parameter(*hg, p_lb);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (params.use_pos)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (pretrained.size()>0)
      t2l = parameter(*hg, p_t2l); 
    state_lstm.new_graph(*hg);
    state_lstm.start_new_sequence();
    //state_lstm.start_new_sequence({zeroes(*hg, {params.state_hidden_dim}), state_start});
    
    Expression sent_start = parameter(*hg, p_sent_start);
    Expression sent_end = parameter(*hg, p_sent_end);
    //stack attention
    Expression s_input2att = parameter(*hg, p_s_input2att);
    Expression s_h2att = parameter(*hg, p_s_h2att);
    Expression s_attbias = parameter(*hg, p_s_attbias);
    Expression s_att2attexp = parameter(*hg, p_s_att2attexp);
    Expression s_att2combo = parameter(*hg, p_s_att2combo);

    //buffer attention
    Expression b_input2att = parameter(*hg, p_b_input2att);
    Expression b_h2att = parameter(*hg, p_b_h2att);
    Expression b_attbias = parameter(*hg, p_b_attbias);
    Expression b_att2attexp = parameter(*hg, p_b_att2attexp);
    Expression b_att2combo = parameter(*hg, p_b_att2combo);

    Expression h2combo = parameter(*hg, p_h2combo);
    Expression combobias = parameter(*hg, p_combobias);
    Expression combo2rt = parameter(*hg, p_combo2rt);
    Expression rtbias = parameter(*hg, p_rtbias);
    vector<Expression> input_expr;
    

/*    if (train) {
      l2rbuilder.set_dropout(params.pdrop);
      r2lbuilder.set_dropout(params.pdrop);
      state_lstm.set_dropout(params.pdrop);
    } else {
      l2rbuilder.disable_dropout();
      r2lbuilder.disable_dropout();
      state_lstm.disable_dropout();
    }
*/
    for (unsigned i = 0; i < sent.size(); ++i) {
      int wordid = sent.raw[i]; // this will be equal to unk at dev/test
      if (train && singletons.size() > wordid && singletons[wordid] && rand01() > params.unk_prob)
          wordid = sent.unk[i];
      if (!train)
	  wordid = sent.unk[i];

      Expression w =lookup(*hg, p_w, wordid);
      if(train) w = dropout(w, params.pdrop);
      vector<Expression> args = {lb, w2l, w}; // learn embeddings
      if (params.use_pos) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sent.pos[i]);
	if(train) p = dropout(p, params.pdrop);
        args.push_back(p2l);
        args.push_back(p);
      }
      if (pretrained.size() > 0 &&  pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, sent.lc[i]);
	if(train) t = dropout(t, params.pdrop);
        args.push_back(t2l);
        args.push_back(t);
      }
      input_expr.push_back(rectify(affine_transform(args)));
    }
if(params.debug)	std::cerr<<"lookup table ok\n";
    vector<Expression> l2r(sent.size());
    vector<Expression> r2l(sent.size());
    Expression l2r_s = l2rbuilder.add_input(sent_start);
    Expression r2l_e = r2lbuilder.add_input(sent_end);
    for (unsigned i = 0; i < sent.size(); ++i) {
      l2r[i] = l2rbuilder.add_input(input_expr[i]);
      r2l[sent.size() - 1 - i] = r2lbuilder.add_input(input_expr[sent.size()-1-i]);
    }
    Expression l2r_e = l2rbuilder.add_input(sent_end);
    Expression r2l_s = r2lbuilder.add_input(sent_start);
    vector<Expression> input(sent.size());
    for (unsigned i = 0; i < sent.size(); ++i) {
      input[i] = concatenate({l2r[i],r2l[i]});
    }
    Expression sent_start_expr = concatenate({l2r_s, r2l_s});
    Expression sent_end_expr = concatenate({l2r_e, r2l_e});
if(params.debug)	std::cerr<<"bilstm ok\n";
    // dummy symbol to represent the empty buffer

    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right

    // in the discriminative model, here we set up the buffer contents
    // dummy symbol to represent the empty buffer
    bufferi[0] = -999;
    vector<int> stacki; // position of words in the sentence of head of subtree
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned stack_buffer_split = 0;
    unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    int nopen_parens = 0;
    char prev_a = '0';
    unsigned unary = 0;//ADDED

    vector<Expression> l2rhc = l2rbuilder.final_s();
    vector<Expression> r2lhc = r2lbuilder.final_s();

    vector<Expression> initc;
    //for(unsigned i = 0; i < params.layers; i ++){
      initc.push_back(concatenate({l2rhc.back(),r2lhc.back()}));
    //}

    //for(unsigned i = 0; i < params.layers; i ++){
      initc.push_back(zeroes(*hg, {params.bilstm_hidden_dim*2}));
    //}
    state_lstm.start_new_sequence(initc);

    while(true){
   	if(prev_a == 'T') break;
      // get list of possible actions for the current parser state
if(params.debug) cerr<< "action_count " << action_count <<"\n";
	current_valid_actions.clear();
if(params.debug) cerr<< "nopen_parens: "<<nopen_parens<<"\n";
      for (auto a : possible_actions) {
        if (IsActionForbidden_Discriminative(adict.convert(a), prev_a, bufferi.size(), stacki.size(), nopen_parens,unary))
          continue;
        current_valid_actions.push_back(a);
      }
      //cerr << "valid actions = " << current_valid_actions.size() << endl;
if(params.debug){
        cerr <<"current_valid_actions: "<<current_valid_actions.size()<<" :";
        for(unsigned i = 0; i < current_valid_actions.size(); i ++){
                cerr<<adict.convert(current_valid_actions[i])<<" ";
        }
        cerr <<"\n";
}

      Expression prev_h = state_lstm.final_h()[0];
      vector<Expression> s_att;
      vector<Expression> s_input;
      s_att.push_back(tanh(affine_transform({s_attbias, s_input2att, sent_start_expr, s_h2att, prev_h})));
      s_input.push_back(sent_start_expr);
      for(unsigned i = 0; i < stack_buffer_split; i ++){
        s_att.push_back(tanh(affine_transform({s_attbias, s_input2att, input[i], s_h2att, prev_h})));
        s_input.push_back(input[i]);
      }
      Expression s_att_col = transpose(concatenate_cols(s_att));
      Expression s_attexp = softmax(s_att_col * s_att2attexp);
if(params.debug){
	auto s_see = as_vector(hg->incremental_forward(s_attexp));
	for(unsigned i = 0; i < s_see.size(); i ++){
		cerr<<s_see[i]<<" ";
	}
	cerr<<"\n";
}
      Expression s_input_col = concatenate_cols(s_input);
      Expression s_att_pool = s_input_col * s_attexp;

      vector<Expression> b_att;
      vector<Expression> b_input;
      for(unsigned i = stack_buffer_split; i < sent.size(); i ++){
        b_att.push_back(tanh(affine_transform({b_attbias, b_input2att, input[i], b_h2att, prev_h})));
        b_input.push_back(input[i]);
      }
      b_att.push_back(tanh(affine_transform({b_attbias, b_input2att, sent_end_expr, b_h2att, prev_h})));
      b_input.push_back(sent_end_expr);
      Expression b_att_col = transpose(concatenate_cols(b_att));
      Expression b_attexp = softmax(b_att_col * b_att2attexp);
if(params.debug){
	auto b_see = as_vector(hg->incremental_forward(b_attexp));
	for(unsigned i = 0; i < b_see.size(); i ++){
		cerr<<b_see[i]<<" ";
	}
	cerr<<"\n";
}
      Expression b_input_col = concatenate_cols(b_input);
      Expression b_att_pool = b_input_col * b_attexp;

if(params.debug)	std::cerr<<"attention ok\n";
      Expression combo = affine_transform({combobias, h2combo, prev_h, s_att2combo, s_att_pool, b_att2combo, b_att_pool});
      Expression n_combo = rectify(combo);
      Expression r_t = affine_transform({rtbias, combo2rt, n_combo});
if(params.debug)	std::cerr<<"to action layer ok\n";

      if (sample) r_t = r_t * params.alpha;
      // adist = log_softmax(r_t, current_valid_actions)
      Expression r_t_s = select_rows(r_t, current_valid_actions);
      Expression adiste = log_softmax(r_t_s);
      vector<float> adist = as_vector(hg->incremental_forward(adiste));
      double best_score = adist[0];
      unsigned model_action = current_valid_actions[0];
      if (sample) {
        double p = rand01();
        assert(current_valid_actions.size() > 0);
        unsigned w = 0;
        for (; w < current_valid_actions.size(); ++w) {
          p -= exp(adist[w]);
          if (p < 0.0) { break; }
        }
        if (w == current_valid_actions.size()) w--;
        model_action = current_valid_actions[w];
      } else { // max
        for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
          if (adist[i] > best_score) {
            best_score = adist[i];
            model_action = current_valid_actions[i];
          }
        }
      }
      unsigned action = model_action;
      if (train) {  // if we have reference actions (for training) use the reference action
        if (action_count >= correct_actions.size()) {
          cerr << "Correct action list exhausted, but not in final parser state.\n";
          abort();
        }
        action = correct_actions[action_count];
        if (model_action == action) { (*right)++; }
      } else {
        //cerr << "Chosen action: " << adict.convert(action) << endl;
      }
      //cerr << "prob ="; for (unsigned i = 0; i < adist.size(); ++i) { cerr << ' ' << adict.convert(i) << ':' << adist[i]; }
      //cerr << endl;
      ++action_count;
      unsigned w = 0;
      for(;w< current_valid_actions.size(); w++){
        if(current_valid_actions[w] == action) break;
      } 
      assert(w != current_valid_actions.size());
    
      log_probs.push_back(pick(adiste, w));
      //results.push_back(action);

      // add current action to action LSTM
      //Expression actione = lookup(*hg, p_a, action);
      //state_lstm.add_input(concatenate({actione,s_att_pool,b_att_pool}));





      apply_action(hg, state_lstm, s_att_pool, b_att_pool, bufferi, stacki, results, action, is_open_paren, stack_buffer_split, nopen_parens, nt_count, prev_a, unary);



    }


    if (train && action_count != correct_actions.size()) {
      cerr << "Unexecuted actions remain but final state reached!\n";
      abort();
    }
    assert(stacki.size() == 2);
    assert(bufferi.size() == 1);
    Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return tot_neglogprob;
  }


// *** if correct_actions is empty, this runs greedy decoding ***
// returns parse actions for input sentence (in training just returns the reference)
// this lets us use pretrained embeddings, when available, for words that were OOV in the
// parser training data
// set sample=true to sample rather than max
Expression log_prob_parser_beam(ComputationGraph* hg,
                     const parser::Sentence& sent,
                     const vector<int>& correct_actions,
                     double *right,
		     vector<unsigned>& results,
		     bool train,
		     unsigned beam_size,//NEW
                     bool sample = false) {

    l2rbuilder.new_graph(*hg);
    r2lbuilder.new_graph(*hg);
    l2rbuilder.start_new_sequence();
    r2lbuilder.start_new_sequence();
    // variables in the computation graph representing the parameters
    Expression lb = parameter(*hg, p_lb);
    Expression w2l = parameter(*hg, p_w2l);
    Expression p2l;
    if (params.use_pos)
      p2l = parameter(*hg, p_p2l);
    Expression t2l;
    if (pretrained.size()>0)
      t2l = parameter(*hg, p_t2l);
    state_lstm.new_graph(*hg);
    state_lstm.start_new_sequence();
    //state_lstm.start_new_sequence({zeroes(*hg, {params.state_hidden_dim}), state_start});

    Expression sent_start = parameter(*hg, p_sent_start);
    Expression sent_end = parameter(*hg, p_sent_end);
    //stack attention
    Expression s_input2att = parameter(*hg, p_s_input2att);
    Expression s_h2att = parameter(*hg, p_s_h2att);
    Expression s_attbias = parameter(*hg, p_s_attbias);
    Expression s_att2attexp = parameter(*hg, p_s_att2attexp);
    Expression s_att2combo = parameter(*hg, p_s_att2combo);

    //buffer attention
    Expression b_input2att = parameter(*hg, p_b_input2att);
    Expression b_h2att = parameter(*hg, p_b_h2att);
    Expression b_attbias = parameter(*hg, p_b_attbias);
    Expression b_att2attexp = parameter(*hg, p_b_att2attexp);
    Expression b_att2combo = parameter(*hg, p_b_att2combo);

    Expression h2combo = parameter(*hg, p_h2combo);
    Expression combobias = parameter(*hg, p_combobias);
    Expression combo2rt = parameter(*hg, p_combo2rt);
    Expression rtbias = parameter(*hg, p_rtbias);
    vector<Expression> input_expr;


    //ADDED
//    Expression s_att_pool;
//    Expression b_att_pool;


/*    if (train) {
      l2rbuilder.set_dropout(params.pdrop);
      r2lbuilder.set_dropout(params.pdrop);
      state_lstm.set_dropout(params.pdrop);
    } else {
      l2rbuilder.disable_dropout();
      r2lbuilder.disable_dropout();
      state_lstm.disable_dropout();
    }
*/
    for (unsigned i = 0; i < sent.size(); ++i) {
      int wordid = sent.raw[i]; // this will be equal to unk at dev/test
      if (train && singletons.size() > wordid && singletons[wordid] && rand01() > params.unk_prob)
          wordid = sent.unk[i];
      if (!train)
	  wordid = sent.unk[i];

      Expression w =lookup(*hg, p_w, wordid);
      if(train) w = dropout(w, params.pdrop);
      vector<Expression> args = {lb, w2l, w}; // learn embeddings
      if (params.use_pos) { // learn POS tag?
        Expression p = lookup(*hg, p_p, sent.pos[i]);
	if(train) p = dropout(p, params.pdrop);
        args.push_back(p2l);
        args.push_back(p);
      }
      if (pretrained.size() > 0 &&  pretrained.count(sent.lc[i])) {  // include fixed pretrained vectors?
        Expression t = const_lookup(*hg, p_t, sent.lc[i]);
	if(train) t = dropout(t, params.pdrop);
        args.push_back(t2l);
        args.push_back(t);
      }
      input_expr.push_back(rectify(affine_transform(args)));
    }
//if(params.debug)	std::cerr<<"lookup table ok\n";
    vector<Expression> l2r(sent.size());
    vector<Expression> r2l(sent.size());
    Expression l2r_s = l2rbuilder.add_input(sent_start);
    Expression r2l_e = r2lbuilder.add_input(sent_end);
    for (unsigned i = 0; i < sent.size(); ++i) {
      l2r[i] = l2rbuilder.add_input(input_expr[i]);
      r2l[sent.size() - 1 - i] = r2lbuilder.add_input(input_expr[sent.size()-1-i]);
    }
    Expression l2r_e = l2rbuilder.add_input(sent_end);
    Expression r2l_s = r2lbuilder.add_input(sent_start);
    vector<Expression> input(sent.size());
    for (unsigned i = 0; i < sent.size(); ++i) {
      input[i] = concatenate({l2r[i],r2l[i]});
    }
    Expression sent_start_expr = concatenate({l2r_s, r2l_s});
    Expression sent_end_expr = concatenate({l2r_e, r2l_e});
//if(params.debug)	std::cerr<<"bilstm ok\n";
    // dummy symbol to represent the empty buffer

    vector<int> bufferi(sent.size() + 1);  // position of the words in the sentence
    // precompute buffer representation from left to right

    // in the discriminative model, here we set up the buffer contents
    // dummy symbol to represent the empty buffer
    bufferi[0] = -999;
    vector<int> stacki; // position of words in the sentence of head of subtree
    stacki.push_back(-999); // not used for anything
    // drive dummy symbol on stack through LSTM
    vector<int> is_open_paren; // -1 if no nonterminal has a parenthesis open, otherwise index of NT
    is_open_paren.push_back(-1); // corresponds to dummy symbol
    vector<Expression> log_probs;
    string rootword;
    unsigned stack_buffer_split = 0;
    //unsigned action_count = 0;  // incremented at each prediction
    unsigned nt_count = 0; // number of times an NT has been introduced
    vector<unsigned> current_valid_actions;
    int nopen_parens = 0;
    char prev_a = '0';
    unsigned unary = 0;//ADDED

   

    vector<Expression> l2rhc = l2rbuilder.final_s();
    vector<Expression> r2lhc = r2lbuilder.final_s();

    vector<Expression> initc;
    //for(unsigned i = 0; i < params.layers; i ++){
      initc.push_back(concatenate({l2rhc.back(),r2lhc.back()}));
    //}

    //for(unsigned i = 0; i < params.layers; i ++){
      initc.push_back(zeroes(*hg, {params.bilstm_hidden_dim*2}));
    //}
    state_lstm.start_new_sequence(initc);

//================================================================================================================================================
        // BEAM SEARCH
//================================================================================================================================================
	   int newcount = 0;
	   int delcount = 0;

	   ParserState* init = new ParserState(); newcount ++;
//	   init->l2rbuilder = l2rbuilder;
//	   init->r2lbuilder = r2lbuilder;
	   init->state_lstm = state_lstm;
	   init->bufferi = bufferi;
	   init->stacki = stacki;
	   init->results = results;
	   init->score = 0;
	   init->gold = true;
	   init->nopen_parens=nopen_parens;
	   init->is_open_paren=is_open_paren;
	   init->prev_a=prev_a;
	   init->unary=unary;
	   init->nt_count=nt_count;
	   init->stack_buffer_split=stack_buffer_split;

	   

	   if (init->stacki.size() ==1 && init->bufferi.size() == 1) { assert(!"bad0"); }

        vector<ParserState*> ongoing; // represents the currently-active beams
        ongoing.push_back(init);
        vector<StepSelect> next_beams; // represents the "next" set of beams, to be used when all current beams are exhausted
        vector<ParserState*> completed; // contains beams that have parsed the whole sentence
        unordered_set<ParserState*> need_to_delete;
        need_to_delete.insert(init);

        double beam_acceptance_percentage;
        if (DYNAMIC_BEAM) { beam_acceptance_percentage = log((100-beam_size)/100.0); beam_size = 10; }
        unsigned active_beams = beam_size; // counts the number of incomplete beams we still need to process
        ParserState* gold_parse = init;
        unsigned action_count = 0;  // incremented at each prediction
        bool full_gold_found = false;


    
        while (completed.size() < beam_size){
            if (ongoing.size() == 0) { // if we've run out of beams in the current step, start on the next one
//                auto step_end = std::chrono::high_resolution_clock::now();
//                double dur = std::chrono::duration<double, std::milli>(step_end - step_start).count();
//                step_start = step_end;

            	

                if (next_beams.size() == 0) {
                    // Sometimes, we have completed all the beams we can, but we set the beam size too high, and there
                    // just aren't enough unique moves to complete more. In that case, we are just done.
                	
                	break;
                }

                // Move the next set to be the current set
                for (StepSelect st : next_beams) {
                    // Create a new ParserState, copying the current one
                    ParserState* ns = new ParserState(); newcount++;
                    need_to_delete.insert(ns); // this prevents memory leaks
                    *ns = *(st.source);

                    // Update the score
                    ns->score += st.action.score;

                    // update the goldness
                    if (train && (!ns->gold || st.action.val != correct_actions[action_count])){
                    
                    	ns->gold = false;
                    }

                    // action_log_prob = pick(adist, action)
                    ns->log_probs.push_back(st.action.log_prob);
                    // do action

   

                    apply_action_to_state(hg, ns, st.action.val);
                    ongoing.push_back(ns);
                }
                next_beams.clear();
                ++action_count;
   
                // if we have reference actions (for training), and are doing early-update,
                // check whether we need to cut off parsing of the sentence
                if (train) {

                    bool gold_in_beam = full_gold_found;
                    for (ParserState* ps : ongoing) {

                    	if (ps->gold) {
                            gold_in_beam = true;
                            break;
                        }
                    }
                    if (!gold_in_beam) {
                        Action gold_action = gold_parse->next_gold_action;

                        gold_parse->score += gold_action.score;
                        // action_log_prob = pick(adist, action)
                        gold_parse->log_probs.push_back(gold_action.log_prob);
                        // is this necessary?
                        apply_action_to_state(hg, gold_parse, gold_action.val );
             
                        break;
                    }
                }
            }

             while (ongoing.size() != 0) {

					// get the state of a beam, and remove that beam from ongoing (because it has been processed)
					ParserState *cur = ongoing.back();
					need_to_delete.insert(cur); // this prevents memory leaks
					ongoing.pop_back();
//					cerr << "sc2 " << ongoing.front()->score << "\n";

					// check whether the current beam is completed
					//if (cur->stacki.size() == 2 && cur->bufferi.size() == 1) {
					if (cur->prev_a == 'T') {
					
						completed.push_back(cur);
						if (cur->gold) {
							gold_parse = cur;
							full_gold_found = true;
						}
						--active_beams;
						if (completed.size() == beam_size)
							break; // we have completed all the beams we need, so just end here
						continue;
					}

			
					// Since we have now confirmed that the beam is not complete, we want to generate all possible actions to
					// take from here, and keep the best states for the next beam set
					getNextBeamsArgs nba{s_attbias, s_input2att, sent_start_expr, s_h2att, b_attbias, b_input2att, sent_end_expr,
						b_h2att, s_att2combo, b_att2combo, h2combo, combobias, combo2rt, rtbias, s_att2attexp, b_att2attexp,
						//s_att_pool, b_att_pool,
						train, correct_actions, action_count, sent.size(), input};

						vector<StepSelect> potential_next_beams;
				       getNextBeams(cur, &potential_next_beams,
									 hg,
									 nba,
									 gold_parse);
						next_beams.insert(next_beams.end(), potential_next_beams.begin(), potential_next_beams.end());

						potential_next_beams.clear();//ADDED

            }


            // cull down next_beams to just keep the best beams
            // keep the next_beams sorted
            sort(next_beams.begin(), next_beams.end(), StepSelectCompare());

            if (DYNAMIC_BEAM) {
                        		if (next_beams.size() > 0) {
            						while ((next_beams.back()).total_score <
            							   (next_beams.front()).total_score + beam_acceptance_percentage ||
            							   next_beams.size() > beam_size) {
            							next_beams.pop_back();
            						}
                        		}
             } else {
                            while (next_beams.size() > active_beams) {
                                next_beams.pop_back();
                            }
             }



        }


	    auto got_answers = std::chrono::high_resolution_clock::now();
        // if we are training, just use the gold one
        if (train) {
//            l2rbuilder = gold_parse->l2rbuilder;
//            r2lbuilder = gold_parse->r2lbuilder;



            state_lstm = gold_parse->state_lstm;
            stacki = gold_parse->stacki;
            bufferi = gold_parse->bufferi;
            results = gold_parse->results;
            nopen_parens = gold_parse->nopen_parens;
            nt_count = gold_parse->nt_count;
	        is_open_paren = gold_parse->is_open_paren;
            prev_a = gold_parse->prev_a;
            unary = gold_parse->unary;
	        stack_buffer_split = gold_parse->stack_buffer_split;
//            s_att_pool = gold_parse->s_att_pool;
//            b_att_pool = gold_parse->b_att_pool;
            log_probs = gold_parse->log_probs;



   
            assert(results.size() <= correct_actions.size());



//            if(results->size() <= correct_actions.size()){
				for (unsigned i = 0; i < results.size(); i++) {
					unsigned a=results.at(i);
					if (correct_actions[i] == a ) { (*right)++; }
					if(params.debug)cerr << adict.convert(a)<<", ";
				}
				  if(params.debug)cerr <<"\n";
//            }else{
//            	for (unsigned i = 0; i < correct_actions.size(); i++) {
//            						unsigned a=results->at(i);
//            						if (correct_actions[i] == a ) { (*right)++; }
//            	}
//            }



        } else { // if we don't have answers, just take the results from the best beam
            sort(completed.begin(), completed.end(), ParserStatePointerCompare());

//            l2rbuilder = completed.front()->l2rbuilder;
//            r2lbuilder = completed.front()->r2lbuilder;


            state_lstm = completed.front()->state_lstm;
            stacki = completed.front()->stacki;
            bufferi = completed.front()->bufferi;
            results = completed.front()->results;
            nopen_parens = completed.front()->nopen_parens;
            nt_count = completed.front()->nt_count;
	    is_open_paren = completed.front()->is_open_paren;
            prev_a = completed.front()->prev_a;
            unary = completed.front()->unary;
	    stack_buffer_split = completed.front()->stack_buffer_split;
//            s_att_pool = completed.front()->s_att_pool;
//            b_att_pool = completed.front()->b_att_pool;
            log_probs = completed.front()->log_probs;

            assert(stacki.size() == 2);
            assert(bufferi.size() == 1);

            auto overall_end = std::chrono::high_resolution_clock::now();
        }




        //Expression intermediate_loss;

        //intermediate_loss = -sum(log_probs);




        // prevents memory leaks
        ongoing.clear();
        next_beams.clear();
        completed.clear();
        for (ParserState* ps: need_to_delete) {delete ps; delcount++;}
        need_to_delete.clear();



        if(params.debug)exit(0);

	    //Expression tot_neglogprob;
        //tot_neglogprob = intermediate_loss;
        //assert(tot_neglogprob.pg != nullptr);
        //return tot_neglogprob;
    
    
     Expression tot_neglogprob = -sum(log_probs);
    assert(tot_neglogprob.pg != nullptr);
    return tot_neglogprob;

}



static void prune(vector<ParserState>& pq, unsigned k) {
  if (pq.size() == 1) return;
  if (k > pq.size()) k = pq.size();
  partial_sort(pq.begin(), pq.begin() + k, pq.end(), ParserStateCompare());
  pq.resize(k);
  reverse(pq.begin(), pq.end());
  //cerr << "PRUNE\n";
  //for (unsigned i = 0; i < pq.size(); ++i) {
  //  cerr << pq[i].score << endl;
  //}
}

static bool all_complete(const vector<ParserState>& pq) {
  for (auto& ps : pq) if (!ps.complete) return false;
  return true;
}

};


void signal_callback_handler(int /* signum */) {
  if (requested_stop) {
    cerr << "\nReceived SIGINT again, quitting.\n";
    _exit(1);
  }
  cerr << "\nReceived SIGINT terminating optimization early...\n";
  requested_stop = true;
}

int main(int argc, char** argv) {
  DynetParams dynet_params = extract_dynet_params(argc, argv);
  dynet_params.random_seed = 1989121013;
  dynet::initialize(dynet_params);
  cerr << "COMMAND:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) cerr << ' ' << argv[i];
  cerr << endl;
  unsigned status_every_i_iterations = 100;


  get_args(argc, argv, params);

  params.state_input_dim = params.action_dim + params.bilstm_hidden_dim*4;
  params.state_hidden_dim = params.bilstm_hidden_dim * 2;

  cerr << "Unknown word strategy: ";
  if (params.unk_strategy) {
    cerr << "STOCHASTIC REPLACEMENT\n";
  } else {
    abort();
  }
  assert(params.unk_prob >= 0.); assert(params.unk_prob <= 1.);
  ostringstream os;
  os << "parser"
     << '_' << params.layers
     << '_' << params.input_dim
     << '_' << params.action_dim
     << '_' << params.pos_dim
     << '_' << params.rel_dim
     << '_' << params.bilstm_input_dim
     << '_' << params.bilstm_hidden_dim
     << '_' << params.attention_hidden_dim
     << "-pid" << getpid() <<"_beam_"<<beam<<".params";
  const string fname = os.str();
  cerr << "Writing parameters to file: " << fname << endl;

//=====================================================================================================
  
  parser::TopDownOracle corpus(&termdict, &adict, &posdict, &ntermdict);
  corpus.load_oracle(params.train_file, true);
  corpus.load_bdata(params.bracketed_file);

  if (params.words_file != "") {
    cerr << "Loading from " << params.words_file << " with" << params.pretrained_dim << " dimensions\n";
    ifstream in(params.words_file.c_str());
    string line;
    getline(in, line);
    vector<float> v(params.pretrained_dim, 0);
    string word;
    while (getline(in, line)) {
      istringstream lin(line);
      lin >> word;
      for (unsigned i = 0; i < params.pretrained_dim; ++i) lin >> v[i];
      unsigned id = termdict.convert(word);
      pretrained[id] = v;
    }
  }

  // freeze dictionaries so we don't accidentaly load OOVs
  termdict.freeze();
  termdict.set_unk("UNK"); // we don't actually expect to use this often
  adict.freeze();
  ntermdict.freeze();
  posdict.freeze();

  {  // compute the singletons in the parser's training data
    unordered_map<unsigned, unsigned> counts;
    for (auto& sent : corpus.sents)
      for (auto word : sent.raw) counts[word]++;
    singletons.resize(termdict.size(), false);
    for (auto wc : counts)
      if (wc.second == 1) singletons[wc.first] = true;
  }

  for (unsigned i = 0; i < adict.size(); ++i) {
    const string& a = adict.convert(i);
    if (a[0] != 'N') continue;
    size_t start = a.find('(') + 1;
    size_t end = a.rfind(')');
    int nt = ntermdict.convert(a.substr(start, end - start));
    action2NTindex[i] = nt;
  }

  NT_SIZE = ntermdict.size()+10;
  POS_SIZE = posdict.size()+10;
  VOCAB_SIZE = termdict.size()+10;
  ACTION_SIZE = adict.size()+10;


  for(unsigned i = 0; i < adict.size(); ++i) possible_actions.push_back(i);

  parser::TopDownOracle dev_corpus(&termdict, &adict, &posdict, &ntermdict);
  parser::TopDownOracle test_corpus(&termdict, &adict, &posdict, &ntermdict);
  if(params.dev_file != "") dev_corpus.load_oracle(params.dev_file, false);
  if(params.test_file != "") test_corpus.load_oracle(params.test_file, false);
  
//============================================================================================================

  Model model;
  ParserBuilder parser(&model, pretrained);
  if (params.model_file != "") {
    TextFileLoader loader(params.model_file);
    loader.populate(model);
  }

  //TRAINING
  if (params.train) {
    signal(SIGINT, signal_callback_handler);

    Trainer* sgd = NULL;
    unsigned method = params.train_methods;
    if(method == 0){
    	cerr << "SGTD selected"<< endl;
        sgd = new SimpleSGDTrainer(model,0.1, 0.1);
        if(beam==0){
        	cerr << "Eta decay 0.05"<< endl;
        	sgd->eta_decay = 0.05;
		}else{
			cerr << "Eta decay 0.08"<< endl;
			sgd->eta_decay = 0.08;
		}
    }
    else if(method == 1)
        sgd = new MomentumSGDTrainer(model,0.01, 0.9, 0.1);
    else if(method == 2){
        sgd = new AdagradTrainer(model);
        sgd->clipping_enabled = false;
    }
    else if(method == 3){
        sgd = new AdamTrainer(model);
        sgd->clipping_enabled = false;
    }

    vector<unsigned> order(corpus.sents.size());
    for (unsigned i = 0; i < corpus.sents.size(); ++i)
      order[i] = i;
    double tot_seen = 0;
    status_every_i_iterations = min((int)status_every_i_iterations, (int)corpus.sents.size());
    unsigned si = corpus.sents.size();
    cerr << "NUMBER OF TRAINING SENTENCES: " << corpus.sents.size() << endl;
    unsigned trs = 0;
    unsigned words = 0;
    double right = 0;
    double llh = 0;
    bool first = true;
    int iter = -1;
    double best_dev_err = 9e99;
    double bestf1=0.0;
    //cerr << "TRAINING STARTED AT: " << put_time(localtime(&time_start), "%c %Z") << endl;
    while(!requested_stop) {
      ++iter;
      auto time_start = chrono::system_clock::now();
      for (unsigned sii = 0; sii < status_every_i_iterations; ++sii) {
           if (si == corpus.sents.size()) {
             si = 0;
             if (first) { first = false; } else { sgd->update_epoch(); }
             cerr << "**SHUFFLE\n";
             random_shuffle(order.begin(), order.end());
           }
           tot_seen += 1;
           auto& sentence = corpus.sents[order[si]];
	   const vector<int>& actions=corpus.actions[order[si]];
	  
           ComputationGraph hg;
           vector<unsigned> pred;//ADDED

//           for(int i=0;i<actions.size();i++)
//           {
//        	   cerr<<">>"<<adict.convert(actions[i]);
//           }
//           cerr<<endl;

//           Expression nll = parser.log_prob_parser(&hg,sentence,actions,&right,&pred,true,false);
           Expression nll;
           if (beam == 0){
        	   nll = parser.log_prob_parser(&hg,sentence,actions,&right,pred,true,false);
           }else{
               
        	   nll = parser.log_prob_parser_beam(&hg,sentence,actions,&right,pred,true,beam,false);
           }
           double lp = as_scalar(hg.incremental_forward(nll));
           if (lp < 0) {
             cerr << "Log prob < 0 on sentence " << order[si] << ": lp=" << lp << endl;
             assert(lp >= 0.0);
           }
           hg.backward(nll);
           sgd->update(1.0);
           llh += lp;
           ++si;
           trs += actions.size();
           words += sentence.size();
           pred.clear();
      }
      sgd->status();

      auto time_now = chrono::system_clock::now();
      auto dur = chrono::duration_cast<chrono::milliseconds>(time_now - time_start);
      cerr << "update #" << iter << " (epoch " << (tot_seen / corpus.sents.size()) <<")"
	   << " per-action-ppl: " << exp(llh / trs) 
	   << " per-input-ppl: " << exp(llh / words) 
	   << " per-sent-ppl: " << exp(llh / status_every_i_iterations) 
           << " err: " << (trs - right) / trs
	   << " [" << dur.count() / (double)status_every_i_iterations << "ms per instance]" << "   Memory Usage: " << getMemoryUsage() <<endl;
      //cerr << "   Memory Usage: " << getMemoryUsage() << "\n\n";

      llh = trs = right = words = 0;
      static int logc = 0;
      ++logc;



     if ((beam == 0 && logc % 25 == 1) || logc % 250 == 1) {
     // if (logc % 25 == 1) { // report on dev set
        unsigned dev_size = dev_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;
        ofstream out("dev.out");
        auto t_start = chrono::high_resolution_clock::now();
        for (unsigned sii = 0; sii < dev_size; ++sii) {
           const auto& sentence=dev_corpus.sents[sii];
	   const vector<int>& actions=dev_corpus.actions[sii];
           
	   ComputationGraph hg;
	   vector<unsigned> pred;
//           Expression nll = parser.log_prob_parser(&hg, sentence, actions, &right, &pred, false, false);
	   Expression nll;
	   if (beam == 0){
		   nll = parser.log_prob_parser(&hg, sentence, actions, &right, pred, false, false);
	   }else{
		   nll = parser.log_prob_parser_beam(&hg, sentence, actions, &right, pred, false, beam, false);
	   }
           double lp = as_scalar(hg.incremental_forward(nll));
           llh += lp;

           int ti = 0;
        	/*
                   for (auto a : pred) {
                     if (adict.convert(a)[0] == 'N') {
                       out << '(' << ntermdict.convert(action2NTindex.find(a)->second) << ' ';
                     } else if (adict.convert(a)[0] == 'S') {
                       out << '(' << posdict.convert(sentence.pos[ti]) << ' ' << sentence.surfaces[ti] << ") ";
        		ti ++;
                     } else out << ") ";
                   }*/

         	for (auto a : pred) {
                        //out << adict.convert(a)<<" ";
                   /*     if(adict.Convert(a) == "SHIFT"){
                                out<<" " << posdict.Convert(sentence.pos[ti])<< " " <<sentence.surfaces[ti];
                                ti++;
                        }*/

         	   if (adict.convert(a)[0] == 'N') {
                           out << 'N' << ntermdict.convert(action2NTindex.find(a)->second) << ' ';
                     } else if (adict.convert(a)[0] == 'S') {
                       out << "S ";
        		ti ++;
                     } else if (adict.convert(a)[0] == 'R')
        		{ out << "R ";}
                   }

                   out << endl;
                   trs += actions.size();
        	   dwords += sentence.size();
               pred.clear();
                }

        
        auto t_end = chrono::high_resolution_clock::now();
        out.close();
        double err = (trs - right) / trs;

        std::string command_1="python delin_ino.py dev.out dev.source dev.pos > dev.eval" ;
        	const char* cmd_1=command_1.c_str();
        	cerr<<"de-linearization: "<<system(cmd_1)<<"\n";

        std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" dev.eval > evalbout.txt";
        const char* cmd2=command2.c_str();
        system(cmd2);
        
        std::ifstream evalfile("evalbout.txt");
        std::string lineS;
        std::string brackstr="Bracketing FMeasure";
        double newfmeasure=0.0;
        std::string strfmeasure="";
        bool found=0;
        while (getline(evalfile, lineS) && !newfmeasure){
		if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
			//std::cout<<lineS<<"\n";
			strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;     // alias of size_t

		        newfmeasure = std::stod (strfmeasure,&sz);
			//std::cout<<strfmeasure<<"\n";
		}
        }
        
        cerr << "  **dev (iter=" << iter << " epoch=" << (tot_seen / corpus.size()) << ")\t"
		<<" llh= " << llh
		<<" ppl: " << exp(llh / dwords)
		<<" f1: " << newfmeasure
		<<" err: " << err
	     <<"\t[" << dev_size << " sents in " << chrono::duration<double, milli>(t_end-t_start).count() << " ms]" << "   Memory Usage: " << getMemoryUsage() << endl;
        //cerr << "   Memory Usage: " << getMemoryUsage() << "\n\n";
        if (newfmeasure>bestf1) {
          cerr << "  new best...writing model to " << fname << " ...\n";
          best_dev_err = err;
	  bestf1=newfmeasure;

	  ostringstream part_os;
          part_os << "parser"
                << '_' << params.layers
                << '_' << params.input_dim
                << '_' << params.action_dim
                << '_' << params.pos_dim
                << '_' << params.rel_dim
                << '_' << params.bilstm_input_dim
                << '_' << params.bilstm_hidden_dim
                << '_' << params.attention_hidden_dim
                << "-pid" << getpid()
                << "-part" << (tot_seen/corpus.size()) <<"_beam_"<<beam<< ".params_"<<newfmeasure;
          const string part = part_os.str();

	  TextFileSaver saver("model/"+part);
          saver.save(model);
        }
      }
    }
  } // should do training?
  else{ // do test evaluation
	ofstream out("test.out");
        unsigned test_size = test_corpus.size();
        double llh = 0;
        double trs = 0;
        double right = 0;
        double dwords = 0;

        	auto t_start = chrono::high_resolution_clock::now();
        	for (unsigned sii = 0; sii < test_size; ++sii) {
           		const auto& sentence=test_corpus.sents[sii];
           		const vector<int>& actions=test_corpus.actions[sii];
           		dwords += sentence.size();
           		ComputationGraph hg;
           		vector<unsigned> pred;
//	   		Expression nll = parser.log_prob_parser(&hg,sentence,actions,&right,&pred,false,false);
           		Expression nll;
           		if (beam == 0){
           			nll = parser.log_prob_parser(&hg,sentence,actions,&right,pred,false,false);
           		}else{

           			nll = parser.log_prob_parser_beam(&hg,sentence,actions,&right,pred,false,beam,false);
           		}
	   		double lp = as_scalar(hg.incremental_forward(nll));
           		llh += lp;

           		int ti = 0;
           				/*
           				for (auto a : pred) {
           	             			if (adict.convert(a)[0] == 'N') {
           	               				out << '(' << ntermdict.convert(action2NTindex.find(a)->second) << ' ';
           	             			} else if (adict.convert(a)[0] == 'S') {
           						out << " (" << posdict.convert(sentence.pos[ti]) << " " << sentence.surfaces[ti] << ")";
           						ti ++;
           	             			} else out << ") ";
           	           		}
           				*/
           				for (auto a : pred) {
           						//out << adict.convert(a)<<" ";
           					   /*     if(adict.Convert(a) == "SHIFT"){
           							out<<" " << posdict.Convert(sentence.pos[ti])<< " " <<sentence.surfaces[ti];
           							ti++;
           						}*/

           				 	   if (adict.convert(a)[0] == 'N') {
           						   out << 'N' << ntermdict.convert(action2NTindex.find(a)->second) << ' ';
           					     } else if (adict.convert(a)[0] == 'S') {
           					       out << "S ";
           						ti ++;
           					     } else if (adict.convert(a)[0] == 'R')
           						{ out << "R ";}
           					   }

           					   out << endl;
           	           		trs += actions.size();
                            pred.clear();
           	        	}

        	
        	auto t_end = chrono::high_resolution_clock::now();
        	out.close();
        	double err = (trs - right) / trs;

        	std::string command_1="python delin_ino.py test.out test.source test.pos > test.eval" ;
        		const char* cmd_1=command_1.c_str();
        		cerr<<system(cmd_1)<<"\n";

        	std::string command2="EVALB/evalb -p EVALB/COLLINS.prm "+corpus.devdata+" test.eval > test_evalbout.txt";
        	const char* cmd2=command2.c_str();
		system(cmd2);

		std::ifstream evalfile("test_evalbout.txt");
        	std::string lineS;
        	std::string brackstr="Bracketing FMeasure";
        	double newfmeasure=0.0;
        	std::string strfmeasure="";
        	bool found=0;
        	while (getline(evalfile, lineS) && !newfmeasure){
                	if (lineS.compare(0, brackstr.length(), brackstr) == 0) {
                        //std::cout<<lineS<<"\n";
                        strfmeasure=lineS.substr(lineS.size()-5, lineS.size());
                        std::string::size_type sz;
                        newfmeasure = std::stod (strfmeasure,&sz);
                        //std::cout<<strfmeasure<<"\n";
                	}
        	}

       		cerr<<"F1score: "<<newfmeasure<<"\n";
       		cerr << "Parsed in " << chrono::duration<double, ratio<1>>(t_end-t_start).count() << " segs"<< endl;
    
  }
}
