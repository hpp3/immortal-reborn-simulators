import React, { useEffect, useState } from 'react';
import { Heading, Button, Text, Grid, Box, TextInput, DropButton, Tip } from 'grommet';
import { withTranslation } from 'react-i18next';
import { Configure } from 'grommet-icons';

import { MAX_VALUE, NODES, SHOW_PERCENTAGE } from './components/Constants';

import { Node, NodeType, PerkType } from './components';
import { exactSteiner } from './components/Steiner';

const credits = ['creator', 'dataProvider'];

const AppBar = (props: any) => (
  <Box
    height="xxsmall"
    tag="header"
    direction="row"
    align="center"
    justify="between"
    background="brand"
    pad={{ left: 'medium', right: 'small', vertical: 'small' }}
    elevation="medium"
    style={{ zIndex: '1' }}
    {...props}
  />
);

// const SCALE = 4;
const buildSeparator = '-';
const urlSeparator = '?';
const defaultPoints = 3;

type SummaryType = { [key: string]: { [key: string]: number | undefined | { [key: string]: number } } };

export const TalentSimulator = withTranslation()(({ pageSize, t, i18n }: { pageSize: string; t: any; i18n: any }) => {
  const [initialNodes, setInitialNodes] = useState<NodeType[]>([]);
  const [nodes, setNodes] = useState(initialNodes);
  // const [showAllTooltip, setShowAllTooltip] = useState(false);
  const [showAllTooltip] = useState(false);
  const [totalPoints, setTotalPoints] = useState(0);
  const [summary, setSummary] = useState({} as SummaryType);
  const [searchString, setsearchString] = useState('');
  const [imporBuildString, setImportBuildString] = useState('');
  const [level, setLevel] = useState<number>();
  const [currentX, setCurrentX] = useState(-1200);
  const [currentZoom, setCurrentZoom] = useState(1);
  // const [isMouseHold, setIsMouseHold] = useState(false);
  const [keyNodes, setKeyNodes] = useState<Set<number>>(new Set());
  const [mstNodes, setMstNodes] = useState<Set<number>>(new Set());

  const isChinese = i18n.language === 'cn';
  const isSmall = pageSize === 'small';
  const isLarge = pageSize === 'large';

  useEffect(() => {
    setCurrentZoom(isSmall ? 1 : 0.6);
  }, [isSmall]);

  useEffect(() => {
    const nodes = NODES.map(NODE => {
      NODE.linkedNodesIndexes.forEach(index => {
        if (!NODES[index].linkedNodesIndexes.includes(NODE.id)) {
          console.error(`#Error: missing link from node ${index} to node ${NODE.id}`);
        }
      });
      return {
        id: NODE.id,
        selectedPoints: 0,
        x: NODE.x * 20,
        y: size.height - NODE.y * 20 - 15,
        points: NODE.points,
        type: NODE.type,
        perks: NODE.perks,
        additionalSearchKeywords: NODE.additionalSearchKeywords,
        linkedNodesIndexes: NODE.linkedNodesIndexes,
      };
    });
    setInitialNodes(nodes);
    setNodes(nodes);

    const pathParts = window.location.href.split(urlSeparator);
    const buildString = pathParts[1];
    if (buildString) {
      importBuild(buildString, [...nodes]);
      window.history.pushState('some state', 'some title', pathParts[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // useEffect(() => {
  //   if (pageSize === 'small') {
  //     setShowAllTooltip(true);
  //   }
  // }, [pageSize]);

  const getSize = (maxValue: any) => {
    const width = maxValue.x * 20 + 40;
    const height = maxValue.y * 20 + 20;
    return { width, height, viewBox: `-10 -5 ${width} ${height}` };
  };

  const size = getSize(MAX_VALUE);

  const updateSummary = ({ perks }: NodeType, summary: any, isAdd: boolean, selectedPoints = 1) => {
    const _summary = { ...summary };
    perks.forEach(({ name, fullNameList, type, value = 0, minValue = 0, maxValue = 0, description }: PerkType) => {
        if (!_summary[type]) {
          _summary[type] = {};
        }

        // perks with fullNameList are Base Stats
        if (!!fullNameList) {
          fullNameList.forEach(fullname => {
            _summary[type][fullname] = (_summary[type][fullname] || 0) + (isAdd ? value : -value) * selectedPoints;
          });
        } else if (type.toLocaleLowerCase().includes('stats')) {
          if (value) {
            _summary[type][name] = (_summary[type][name] || 0) + (isAdd ? value : -value) * selectedPoints;
          } else {
            if (!_summary[type][name]) {
              _summary[type][name] = {};
            }
            _summary[type][name].min = (_summary[type][name].min || 0) + (isAdd ? minValue : -minValue) * selectedPoints;
            _summary[type][name].max = (_summary[type][name].max || 0) + (isAdd ? maxValue : -maxValue) * selectedPoints;
          }
        } else {
        _summary[type][name] = isAdd ? description : '';
      }
    });
    return _summary;
  };

		const clearAll = () => {
			setTotalPoints(0);
			setNodes(initialNodes);
			setSummary({});
		};

		const importBuild = (buildString?: string, nodes?: NodeType[]) => {
			const _nodes = initialNodes.length ? [...initialNodes] : nodes || [];
			let _totalPoints = 0;
			let _summary: SummaryType = {};
			(buildString || imporBuildString).split(buildSeparator).forEach(indexStr => {
					const [indexString, selectedPointsString] = indexStr.split('.');
					const index = parseInt(indexString);
					if (!isNaN(index) && index < _nodes.length) {
						const _node = _nodes[index];
						const selectedPoints = parseInt(selectedPointsString) || _node.points;
						_nodes.splice(index, 1, { ..._node, selectedPoints: selectedPoints });
						_totalPoints = _totalPoints + selectedPoints;
						_summary = updateSummary(_node, _summary, true, selectedPoints);
					}
				});
			setTotalPoints(_totalPoints);
			setNodes(_nodes);
			setSummary(_summary);
			setImportBuildString('');
		};

		const getBuild = () => {
			let buildString = '';
			nodes.forEach(node => {
				if (node.selectedPoints) {
					buildString += buildString
						? `${buildSeparator}${node.id}${node.selectedPoints !== node.points ? `.${node.selectedPoints}` : ''}`
						: node.id;
				}
			});
			return buildString;
		};

		const statusPanel = (
			<Box width="5000px" pad={'10px'}>
				<Grid
					fill="horizontal"
					rows={['xxsmall', 'xxsmall', 'xxsmall', 'xxsmall']}
					columns={['22%', '22%', '22%', '22%']}
					gap="small"
					areas={[
						{ name: 'showAll', start: [2, 0], end: [2, 0] },
						{ name: 'resetAll', start: [3, 0], end: [3, 0] },
						{ name: 'importString', start: [0, 1], end: [2, 1] },
						{ name: 'importButton', start: [3, 1], end: [3, 1] },
						{ name: 'currentBuild', start: [0, 2], end: [2, 2] },
						{ name: 'exportButton', start: [3, 2], end: [3, 2] },
						{ name: 'levelText', start: [2, 3], end: [2, 3] },
						{ name: 'level', start: [3, 3], end: [3, 3] },
					]}>
					{/* <Button
          gridArea="showAll"
          primary
          label={showAllTooltip ? t('hideAll') : t('showAll')}
          onClick={() => setShowAllTooltip(!showAllTooltip)}
        /> */}
					<Button gridArea="resetAll" fill={false} primary label={t('resetBuild')} onClick={() => clearAll()} />
					<Box gridArea="importString">
						<TextInput
							placeholder={t('loadBuild')}
							value={imporBuildString}
							onChange={event => {
								setImportBuildString(event.target.value);
							}}
						/>
					</Box>
					<Button gridArea="importButton" fill={false} primary label={t('load')} onClick={() => importBuild()} />
					<Box gridArea="currentBuild">
						<TextInput disabled placeholder={t('currentBuild')} value={getBuild()} />
					</Box>
					<Button
						gridArea="exportButton"
						fill={false}
						primary
						label={t('share')}
						onClick={() => {
							const link = `${window.location.href}${urlSeparator}${getBuild()}`;
							navigator.clipboard.writeText(link).then(() => alert(t('buildCopied', { link })));
						}}
					/>
					<Box gridArea="levelText" alignContent="end" justify="center">
						<Text size="large">{t('level')}</Text>
					</Box>
					<Box gridArea="level">
						<TextInput
							placeholder={t('currentLevel')}
							type="number"
							value={level}
							onChange={event => setLevel(parseInt(event.target.value))}
						/>
					</Box>
				</Grid>

				<Heading size="small">
					{level
						? t('remainPoints', { points: level + defaultPoints - totalPoints })
						: t('reqiredPoints', { totalPoints, Levels: Math.max(totalPoints - defaultPoints, 0) })}
				</Heading>
				<Grid
					fill="vertical"
					rows={isLarge ? ['large'] : ['medium', 'small']}
					columns={isLarge ? ['16%', '16%', '16%', '16%', '16%', '16%'] : ['32%', '32%', '32%']}
					gap="small"
					areas={
						isLarge
							? [
									{ name: 'offensiveStats', start: [0, 0], end: [0, 0] },
									{ name: 'defensiveStats', start: [1, 0], end: [1, 0] },
									{ name: 'passive', start: [2, 0], end: [2, 0] },
									{ name: 'baseStats', start: [3, 0], end: [3, 0] },
									{ name: 'skillLvlStats', start: [4, 0], end: [4, 0] },
									{ name: 'specialStats', start: [5, 0], end: [5, 0] },
								]
							: [
									{ name: 'offensiveStats', start: [0, 0], end: [0, 0] },
									{ name: 'defensiveStats', start: [1, 0], end: [1, 0] },
									{ name: 'passive', start: [2, 0], end: [2, 0] },
									{ name: 'baseStats', start: [0, 1], end: [0, 1] },
									{ name: 'skillLvlStats', start: [1, 1], end: [1, 1] },
									{ name: 'specialStats', start: [2, 1], end: [2, 1] },
								]
					}>
					{['offensiveStats', 'defensiveStats', 'baseStats', 'skillLvlStats', 'passive', 'specialStats'].map(type => {
						return (
							<Box gridArea={type} background="light-5" key={type}>
								<Heading size="small" margin="xsmall">
									{t(type)}
								</Heading>
								{summary[type] && (
									<Box overflow="auto">
										{Object.keys(summary[type]).map(name => {
											const value = summary[type][name];
											let string = '';
											const nameString = t(name);
											let description = '';
											if (typeof value === 'string' && value) {
												string = nameString;
												description = value;
											} else if (typeof value === 'object') {
												const { min, max } = value;
												const valueString = `+${min}~${max}`;
												if (min || max) {
													if (isChinese) {
														string = `${nameString} ${valueString}`;
													} else {
														string = `${valueString} ${nameString}`;
													}
												} else {
													string = '';
												}
											} else if (value) {
												const valueString = SHOW_PERCENTAGE[name] ? `${Math.round(value * 10000) / 100}%` : value;
												if (isChinese) {
													string = `${nameString} +${valueString}`;
												} else {
													string = `+${valueString} ${nameString}`;
												}
											} else {
												string = '';
											}

											return (
												string &&
												(description ? (
													<Tip
														plain
														content={
															<Box direction="row" align="center" pad="none">
																<Box
																	background="white"
																	direction="row"
																	pad="small"
																	round="xsmall"
																	width={{ max: 'small' }}>
																	<Text size="small">{t(description)}</Text>
																</Box>
                              <svg viewBox="0 0 22 22" version="1.1" width="22px" height="22px">
                                <polygon fill="white" points="0 2 12 12 0 22" />
																</svg>
															</Box>
														}
                          dropProps={{ align: { right: 'left' } }}
                          key={name}>
														<Text>{string}</Text>
													</Tip>
												) : (
													<Text key={name}>{string}</Text>
												))
											);
										})}
									</Box>
								)}
							</Box>
						);
					})}
				</Grid>
			</Box>
		);

  const repoProvider = window.location.href.includes('github') ? 'github' : 'gitee';

		// const listeners = {
		//   onMouseMove: (e: any) => {
		//     if (isMouseHold) {
		//       // if (e.touches) { e = e.touches[0]; }
		//       // return {
		//       //   x: (e.clientX - CTM.e) / CTM.a,
		//       //   y: (e.clientY - CTM.f) / CTM.d
		//       // };
		//       setCurrentX(currentX + 10);
		//     }
		//   },
		//   onMouseDown: (e: any) => {
		//     setIsMouseHold(true);
		//   },
		//   onMouseUp: (e: any) => {
		//     setIsMouseHold(false);
		//   },
		//   onMouseLeave: (e: any) => {
		//     setIsMouseHold(false);
		//   },
		// };

		const moveStarMap = (isLeft: boolean) => {
			if ((isLeft && currentX < 100) || (!isLeft && currentX > -2500)) {
				setCurrentX(currentX + 200 * currentZoom * (isLeft ? 1 : -1));
			}
		};

		const zoomStarMap = (isZoomIn: boolean) => {
			if ((isZoomIn && currentZoom < 2) || (!isZoomIn && currentZoom > 0.4)) {
				setCurrentZoom(currentZoom + (isZoomIn ? 0.1 : -0.1));
			}
		};

		type Edge = { from: number; to: number; weight: number };

		function computeSteinerTree(
			keyNodeIds: Set<number>,
			nodes: NodeType[],
		): { mstNodeIds: Set<number>; mstEdges: [number, number][] } {
			if (keyNodeIds.size === 0) return { mstNodeIds: new Set(), mstEdges: [] };

			// Create a temporary node that connects to the first 5 nodes
			const tempNode: NodeType = {
				id: nodes.length,
				x: 24,
				y: 12,
				points: 0,
				type: "basic",
				selectedPoints: 0,
				perks: [{ name: "lck", type: "baseStats", value: 0 }],
				additionalSearchKeywords: "",
				linkedNodesIndexes: [0, 1, 2, 3, 4],
			};

			// Create a deep copy of the nodes array
			const nodesWithTemp = nodes.map((node) => ({
				...node,
				linkedNodesIndexes: [...node.linkedNodesIndexes],
			}));

			// modify the first 5 nodes to connect to this node
			nodesWithTemp[0].linkedNodesIndexes.push(tempNode.id);
			nodesWithTemp[1].linkedNodesIndexes.push(tempNode.id);
			nodesWithTemp[2].linkedNodesIndexes.push(tempNode.id);
			nodesWithTemp[3].linkedNodesIndexes.push(tempNode.id);
			nodesWithTemp[4].linkedNodesIndexes.push(tempNode.id);

			// Add the temporary node to the nodes array
			nodesWithTemp.push(tempNode);

			// Convert Set to array for exactSteiner
			const keyIdsArray = Array.from(keyNodeIds);
			keyIdsArray.push(tempNode.id);

			// Use exactSteiner to compute the optimal tree
			const result = exactSteiner(nodesWithTemp, keyIdsArray);

			// remove the temporary node
			const mstNodeIds = new Set(
				Array.from(result.steinerNodes).filter((id) => id !== tempNode.id),
			);
			const mstEdges = result.steinerEdges.filter(
				([from, to]) => from !== tempNode.id && to !== tempNode.id,
			);

			return {
				mstNodeIds,
				mstEdges,
			};
		}

		// New node click handler for MST/key node mode
		const onNodeClick = (nodeId: number) => {
			// Create new key nodes set
			const newKeyNodes = new Set(keyNodes);
			if (newKeyNodes.has(nodeId)) {
				newKeyNodes.delete(nodeId);
			} else {
				newKeyNodes.add(nodeId);
			}

			// Compute Steiner tree using current nodes state
			const { mstNodeIds } = computeSteinerTree(newKeyNodes, initialNodes);

			// Update all states at once using the computed values
			const updatedNodes = initialNodes.map(node => ({
				...node,
				isKeyNode: newKeyNodes.has(node.id),
				isMSTNode: mstNodeIds.has(node.id),
				selectedPoints: mstNodeIds.has(node.id) ? node.points : 0,
			}));

			// Update all states in a single render cycle
			setKeyNodes(newKeyNodes);
			setMstNodes(mstNodeIds);
			setNodes(updatedNodes);

			// Update total points and summary
			const newTotalPoints = updatedNodes.reduce((sum, node) => sum + node.selectedPoints, 0);
			setTotalPoints(newTotalPoints);

			// Recompute summary
			const newSummary = updatedNodes.reduce((acc, node) => {
				if (node.selectedPoints > 0) {
					return updateSummary(node, acc, true, node.selectedPoints);
				}
				return acc;
			}, {} as SummaryType);
			setSummary(newSummary);
		};

		return (
			<>
				<AppBar>
					<Heading level="3" margin="none">
          {t('title')}
					</Heading>
					<Box direction="row">
						<DropButton
							icon={<Configure />}
							dropContent={statusPanel}
            dropProps={{ align: { top: 'bottom', right: 'right' }, background: 'light-1' }}
						/>
          <Button label={t('language')} onClick={() => i18n.changeLanguage(isChinese ? 'en' : 'cn')} />
						<DropButton
            label={t('askForUpdate')}
							dropContent={
								<Box>
									<Button
                  label={t('forum')}
                  onClick={() => window.open(`https://www.taptap.com/topic/15451647`, '_blank')}
									/>
									<Button
                  label={t('bugReport', { repoProvider })}
										onClick={() =>
                    window.open(`https://${repoProvider}.com/mintyknight/immortal-reborn-simulators/issues`, '_blank')
										}
									/>
								</Box>
							}
            dropProps={{ align: { top: 'bottom', right: 'right' }, background: 'light-1' }}
						/>
						<DropButton
            label={t('credit')}
							dropContent={
								<Box>
                {credits.map(credit => (
										<Text key={credit}>{t(credit)}</Text>
									))}
								</Box>
							}
            dropProps={{ align: { top: 'bottom', right: 'right' }, background: 'light-1' }}
						/>
					</Box>
				</AppBar>
      <Box overflow={{ horizontal: 'hidden' }}>
					<Grid
						fill={true}
          rows={['200%']}
          columns={['99%']}
						gap="small"
          areas={[{ name: 'starMap', start: [0, 0], end: [0, 0] }]}>
						<Box gridArea="starMap" background="light-2">
							<Grid
              rows={['100%']}
              columns={['14%', '4%', '14%', '4%', '28%', '4%', '14%', '4%', '14%']}
								areas={[
                { name: 'left', start: [0, 0], end: [0, 0] },
                { name: 'midLeft', start: [2, 0], end: [2, 0] },
                { name: 'middle', start: [4, 0], end: [4, 0] },
                { name: 'midRight', start: [6, 0], end: [6, 0] },
                { name: 'right', start: [8, 0], end: [8, 0] },
              ]}>
              <Button gridArea="left" primary label="<" onClick={() => moveStarMap(true)} />
              <Button gridArea="right" primary label=">" onClick={() => moveStarMap(false)} />
								<Box gridArea="middle">
									<TextInput
                  placeholder={t('search4Perk')}
										value={searchString}
                  onChange={event => setsearchString(event.target.value)}
									/>
								</Box>
              <Button gridArea="midLeft" primary label="+" onClick={() => zoomStarMap(true)} />
              <Button gridArea="midRight" primary label="-" onClick={() => zoomStarMap(false)} />
							</Grid>
            <svg width={'100%'} viewBox={`0 0 ${500 / currentZoom} ${700 * (currentZoom > 1 ? 1 : currentZoom)}`}>
								<rect x={0} y={0} width="100%" height="100%" fill="Gray" />
              <svg x={currentX} width={'3000'} viewBox={size.viewBox}>
                {nodes.map(node => (
										<Node
											key={node.id}
											{...node}
											nodes={nodes}
											showTooltip={showAllTooltip}
											searchString={searchString}
											isChinese={isChinese}
											isKeyNode={keyNodes.has(node.id)}
											isMSTNode={mstNodes.has(node.id)}
											onNodeClick={() => onNodeClick(node.id)}
										/>
									))}
								</svg>
							</svg>
						</Box>
					</Grid>
				</Box>
			</>
		);
});
