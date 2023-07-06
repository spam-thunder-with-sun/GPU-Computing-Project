clauses_ = [1 0 0 0 0 -1; 1 1 1 0 0 0];
clauses = clauses_';
possible = [
    0 0 0 -1 -1 -1; %no
    1 0 0 0 -1 -1; %si
    0 1 0 -1 0 -1; %si
    1 1 0 0 0 -1; %si
    0 0 1 -1 -1 0; %no
    1 0 1 0 -1 0; %si
    0 1 1 -1 0 0; %no
    1 1 1 0 0 0]; %si

mul = [];
ris = [];

for i = 1 : 8
    vet = possible(i, :);
    tmp = vet * clauses;
    mul = [mul; tmp];
    ris = [ris; sum(tmp)];
end

disp(possible * clauses);