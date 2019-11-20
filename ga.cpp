#include <bits/stdc++.h>
//using namespace std;
double randreal(){
    static std::random_device rnd;     // 非決定的な乱数生成器を生成
    static std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
    static std::uniform_real_distribution<> dist(0.0, 1.0);
    return dist(mt);
}
int randint(int n){
    return std::floor(randreal()*n);
}
int randint(int a,int b){
    return randint(b-a+1)+a;
}
class Individual{
public:
    std::vector<int> gene;
    int kind;
    double evaluation=0;
    Individual(int size,int kind){
        this->kind=kind;
        this->gene=std::vector<int>(size);
        for(auto& g:this->gene){
            g=randint(kind);
        }
    }
    int size(){
        return gene.size();
    }
    Individual(std::vector<int> &&gene,int kind){
        this->gene=gene;
        this->kind=kind;
    }
    Individual onePointCrossOver(const Individual &other){
        std::vector<int> ret = this->gene;
        int r=randint(1,ret.size()-1);
        for(int i=r;i<ret.size();++i){
            ret[i]=other.gene[i];
        }
        return std::move(Individual(std::move(ret),this->kind));
    }
    
    Individual twoPointCrossOver(const Individual &other){
        std::vector<int> ret = this->gene;
        int r1=randint(0,ret.size()-2);
        int r2=randint(r1+1,ret.size()-1);
        for(int i=r1;i<=r2;++i){
            ret[i]=other.gene[i];
        }
        return std::move(Individual(std::move(ret),this->kind));
    }
    Individual uniformalCrossOver(const Individual &other){
        std::vector<int> ret = this->gene;
        for(int i=0;i<ret.size();++i){
            if(randint(2)==0){
                ret[i]=other.gene[i];
            }
        }
        return std::move(Individual(std::move(ret),this->kind));
    }
    void mutation(){
        this->gene[randint(this->size())]=randint(this->kind);
    }
};
class Society{
public:
    std::vector<Individual> individuals;
    int genenum;
    Society(int genenum,int genesize,int kind){
        this->individuals = std::vector<Individual>(genenum,Individual(genesize,kind));
        this->genenum = genenum;
    }
    void clossOver(){
        int s = individuals.size();
        while(individuals.size()<genenum){
            int r1 = randint(s);
            int r2 = randint(s-1);
            if(r2>=r1){
                r2+=1;
            }
            //individuals[r1]=individuals[r2];
            individuals.push_back(individuals[r1].twoPointCrossOver(individuals[r2]));
        }
    }
    void mutation(){
        for(int i=0;i<individuals.size();++i){
            if(randint(100)==0){
                individuals[i].mutation();
            }
        }
    }
    void evaluation(){
        for(auto &ind: this->individuals){
            int sum=0;
            for(auto &g: ind.gene){
                sum+=g;
            }
            ind.evaluation=sum;
        }
    }
    void rankingSelection(){
        sort(individuals.begin(),individuals.end(),[](auto &a,auto &b)->bool{
            return a.evaluation>b.evaluation;
        });
        individuals=std::vector<Individual>(individuals.begin(),individuals.begin()+20);
    }
    void tournamentSelection(){
        int numt = 2;
        for(int i=0;i<numt;++i){
            std::vector<Individual> ind;
            for(int j=0;j<individuals.size()/2;++j){
                if(individuals[j*2].evaluation>individuals[j*2+1].evaluation){
                    ind.push_back(individuals[j*2]);
                }else{
                    ind.push_back(individuals[j*2+1]);
                }
            }
            individuals=ind;
        }
    }
    void rouletteSelection(){
        std::vector<double> probabilitysum(individuals.size());
        double sum=0;
        double min=DBL_MAX;
        double max=DBL_MIN;
        for(int i=0;i<individuals.size();++i){
            double ev=individuals[i].evaluation;
            if(ev<min){
                min=ev;
            }
            if(ev>max){
                max=ev;
            }
        }
        if(min==max){
            max+=1;
        }
        for(int i=0;i<individuals.size();++i){
            double ev=individuals[i].evaluation;
            sum+=(ev-min)/(max-min);
            probabilitysum[i]=sum;
        }
        int selectnum = 20;
        std::vector<Individual> inds;
        for(int i=0;i<selectnum;++i){
            auto selitr = std::lower_bound(probabilitysum.begin(),probabilitysum.end(),randreal()*sum);
            inds.push_back(individuals[std::distance(probabilitysum.begin(),selitr)]);
        }
        individuals=inds;
    }
};

int main(){
    Society society(100,100,2);
    for(int i=0;i<1000;++i){
        society.evaluation();
        //society.tournamentSelection();
        //society.rouletteSelection();
        society.rankingSelection();
        society.clossOver();
        std::cout<<society.individuals[0].evaluation<<std::endl;
        society.mutation();
    }
}