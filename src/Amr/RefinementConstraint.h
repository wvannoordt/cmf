#ifndef REFTYPE_H
#define REFTYPE_H
namespace gTree
{
    namespace RefinementConstraint
    {
        enum RefinementConstraint
        {
            free,
            factor2PartiallyConstrained,
            factor2CompletelyConstrained
        };

        inline static std::string RefinementConstraintStr(int refType)
        {
            switch (refType)
            {
                case RefinementConstraint::free: return "free";
                case RefinementConstraint::factor2PartiallyConstrained: return "factor2PartiallyConstrained";
                case RefinementConstraint::factor2CompletelyConstrained: return "factor2CompletelyConstrained";
            }
            return "";
        }
    }
}
#endif
