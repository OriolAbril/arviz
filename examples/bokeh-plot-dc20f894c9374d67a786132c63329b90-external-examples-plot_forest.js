(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("a65b432a-9cd3-4ac6-bf42-23869875eb56");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'a65b432a-9cd3-4ac6-bf42-23869875eb56' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js": "T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js": "98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js": "89bArO+nlbP3sgakeHjCo1JYxYR5wufVgA3IbUvDY+K7w4zyxJqssu7wVnfeKCq8"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.3.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"4554dd45-ede1-4a8b-ae8c-50592a0bfdbb":{"roots":{"references":[{"attributes":{},"id":"17766","type":"Selection"},{"attributes":{"data_source":{"id":"17653"},"glyph":{"id":"17654"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17655"},"selection_glyph":null,"view":{"id":"17657"}},"id":"17656","type":"GlyphRenderer"},{"attributes":{},"id":"17767","type":"UnionRenderers"},{"attributes":{},"id":"17768","type":"Selection"},{"attributes":{"data":{"x":[-1.9958943490877263,9.311550558002505],"y":[1.65,1.65]},"selected":{"id":"17790"},"selection_policy":{"id":"17789"}},"id":"17693","type":"ColumnDataSource"},{"attributes":{},"id":"17769","type":"UnionRenderers"},{"attributes":{},"id":"17770","type":"Selection"},{"attributes":{"children":[{"id":"17816"},{"id":"17814"}]},"id":"17817","type":"Column"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17650","type":"Line"},{"attributes":{"source":{"id":"17643"}},"id":"17647","type":"CDSView"},{"attributes":{},"id":"17771","type":"UnionRenderers"},{"attributes":{},"id":"17772","type":"Selection"},{"attributes":{"data_source":{"id":"17663"},"glyph":{"id":"17664"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17665"},"selection_glyph":null,"view":{"id":"17667"}},"id":"17666","type":"GlyphRenderer"},{"attributes":{},"id":"17812","type":"Selection"},{"attributes":{"bounds":"auto","end":3.4499999999999997,"min_interval":2,"start":-0.8999999999999999},"id":"17756","type":"DataRange1d"},{"attributes":{},"id":"17773","type":"UnionRenderers"},{"attributes":{},"id":"17774","type":"Selection"},{"attributes":{},"id":"17796","type":"Selection"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17649","type":"Line"},{"attributes":{"data":{"x":[-2.13476994033137,10.008689721107494],"y":[0.3,0.3]},"selected":{"id":"17772"},"selection_policy":{"id":"17771"}},"id":"17648","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"17638"},"glyph":{"id":"17639"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17640"},"selection_glyph":null,"view":{"id":"17642"}},"id":"17641","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17648"}},"id":"17652","type":"CDSView"},{"attributes":{},"id":"17775","type":"UnionRenderers"},{"attributes":{},"id":"17776","type":"Selection"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"17614"},{"id":"17615"},{"id":"17616"},{"id":"17617"},{"id":"17618"},{"id":"17619"},{"id":"17620"},{"id":"17621"}]},"id":"17624","type":"Toolbar"},{"attributes":{"children":[[{"id":"17597"},0,0]]},"id":"17814","type":"GridBox"},{"attributes":{},"id":"17777","type":"UnionRenderers"},{"attributes":{},"id":"17778","type":"Selection"},{"attributes":{"data":{"x":[2.2808674873161356,6.6262125124574265],"y":[0.3,0.3]},"selected":{"id":"17774"},"selection_policy":{"id":"17773"}},"id":"17653","type":"ColumnDataSource"},{"attributes":{"ticks":[0.44999999999999996,2.0999999999999996]},"id":"17757","type":"FixedTicker"},{"attributes":{},"id":"17779","type":"UnionRenderers"},{"attributes":{},"id":"17780","type":"Selection"},{"attributes":{"data":{"x":[-1.4753067334040253,11.208804111875878],"y":[0.6,0.6]},"selected":{"id":"17778"},"selection_policy":{"id":"17777"}},"id":"17663","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"17606"}],"center":[{"id":"17609"},{"id":"17613"}],"left":[{"id":"17610"}],"outline_line_color":null,"output_backend":"webgl","plot_height":435,"plot_width":360,"renderers":[{"id":"17636"},{"id":"17641"},{"id":"17646"},{"id":"17651"},{"id":"17656"},{"id":"17661"},{"id":"17666"},{"id":"17671"},{"id":"17676"},{"id":"17681"},{"id":"17686"},{"id":"17691"},{"id":"17696"},{"id":"17701"},{"id":"17706"},{"id":"17711"},{"id":"17716"},{"id":"17721"},{"id":"17726"},{"id":"17731"},{"id":"17736"},{"id":"17741"},{"id":"17746"},{"id":"17751"}],"title":{"id":"17753"},"toolbar":{"id":"17624"},"toolbar_location":null,"x_range":{"id":"17755"},"x_scale":{"id":"17602"},"y_range":{"id":"17756"},"y_scale":{"id":"17604"}},"id":"17597","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"x":[-1.0798320889339172,10.200853218312133],"y":[0,0]},"selected":{"id":"17766"},"selection_policy":{"id":"17765"}},"id":"17633","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"17633"},"glyph":{"id":"17634"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17635"},"selection_glyph":null,"view":{"id":"17637"}},"id":"17636","type":"GlyphRenderer"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17694","type":"Line"},{"attributes":{"data_source":{"id":"17688"},"glyph":{"id":"17689"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17690"},"selection_glyph":null,"view":{"id":"17692"}},"id":"17691","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"17678"},"glyph":{"id":"17679"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17680"},"selection_glyph":null,"view":{"id":"17682"}},"id":"17681","type":"GlyphRenderer"},{"attributes":{},"id":"17781","type":"UnionRenderers"},{"attributes":{},"id":"17761","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"17690","type":"Circle"},{"attributes":{"data_source":{"id":"17698"},"glyph":{"id":"17699"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17700"},"selection_glyph":null,"view":{"id":"17702"}},"id":"17701","type":"GlyphRenderer"},{"attributes":{},"id":"17782","type":"Selection"},{"attributes":{"data":{},"selected":{"id":"17788"},"selection_policy":{"id":"17787"}},"id":"17688","type":"ColumnDataSource"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"17623","type":"PolyAnnotation"},{"attributes":{"data":{"x":[-2.358040118461496,10.910290467635015],"y":[1.95,1.95]},"selected":{"id":"17796"},"selection_policy":{"id":"17795"}},"id":"17708","type":"ColumnDataSource"},{"attributes":{},"id":"17797","type":"UnionRenderers"},{"attributes":{},"id":"17602","type":"LinearScale"},{"attributes":{"source":{"id":"17688"}},"id":"17692","type":"CDSView"},{"attributes":{},"id":"17798","type":"Selection"},{"attributes":{"data_source":{"id":"17668"},"glyph":{"id":"17669"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17670"},"selection_glyph":null,"view":{"id":"17672"}},"id":"17671","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"17708"},"glyph":{"id":"17709"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17710"},"selection_glyph":null,"view":{"id":"17712"}},"id":"17711","type":"GlyphRenderer"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17699","type":"Line"},{"attributes":{"source":{"id":"17693"}},"id":"17697","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17695","type":"Line"},{"attributes":{},"id":"17783","type":"UnionRenderers"},{"attributes":{"text":"94.0% HDI"},"id":"17753","type":"Title"},{"attributes":{"source":{"id":"17668"}},"id":"17672","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17640","type":"Line"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"17705","type":"Circle"},{"attributes":{},"id":"17784","type":"Selection"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"17674","type":"Circle"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.374380781729923},"y":{"value":1.65}},"id":"17704","type":"Circle"},{"attributes":{"data":{"x":[0.9649316231388204,5.949680273009057],"y":[1.65,1.65]},"selected":{"id":"17792"},"selection_policy":{"id":"17791"}},"id":"17698","type":"ColumnDataSource"},{"attributes":{},"id":"17799","type":"UnionRenderers"},{"attributes":{"source":{"id":"17698"}},"id":"17702","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17700","type":"Line"},{"attributes":{},"id":"17800","type":"Selection"},{"attributes":{"data":{"x":[-2.2641273152868076,10.475301593619335],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"17784"},"selection_policy":{"id":"17783"}},"id":"17678","type":"ColumnDataSource"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17709","type":"Line"},{"attributes":{"data_source":{"id":"17703"},"glyph":{"id":"17704"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17705"},"selection_glyph":null,"view":{"id":"17707"}},"id":"17706","type":"GlyphRenderer"},{"attributes":{"data_source":{"id":"17713"},"glyph":{"id":"17714"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17715"},"selection_glyph":null,"view":{"id":"17717"}},"id":"17716","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"17644","type":"Circle"},{"attributes":{"data":{},"selected":{"id":"17794"},"selection_policy":{"id":"17793"}},"id":"17703","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"17648"},"glyph":{"id":"17649"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17650"},"selection_glyph":null,"view":{"id":"17652"}},"id":"17651","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17655","type":"Line"},{"attributes":{"data":{"x":[-2.061896746944455,11.515205993744095],"y":[2.25,2.25]},"selected":{"id":"17802"},"selection_policy":{"id":"17801"}},"id":"17723","type":"ColumnDataSource"},{"attributes":{"source":{"id":"17703"}},"id":"17707","type":"CDSView"},{"attributes":{},"id":"17785","type":"UnionRenderers"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"17622","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17670","type":"Line"},{"attributes":{},"id":"17765","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"17723"},"glyph":{"id":"17724"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17725"},"selection_glyph":null,"view":{"id":"17727"}},"id":"17726","type":"GlyphRenderer"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17679","type":"Line"},{"attributes":{},"id":"17786","type":"Selection"},{"attributes":{"source":{"id":"17638"}},"id":"17642","type":"CDSView"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17714","type":"Line"},{"attributes":{},"id":"17801","type":"UnionRenderers"},{"attributes":{"data":{"x":[2.454317217804662,6.859709821773684],"y":[0,0]},"selected":{"id":"17768"},"selection_policy":{"id":"17767"}},"id":"17638","type":"ColumnDataSource"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17634","type":"Line"},{"attributes":{"source":{"id":"17708"}},"id":"17712","type":"CDSView"},{"attributes":{},"id":"17802","type":"Selection"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17654","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17710","type":"Line"},{"attributes":{"source":{"id":"17633"}},"id":"17637","type":"CDSView"},{"attributes":{"source":{"id":"17663"}},"id":"17667","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"17720","type":"Circle"},{"attributes":{"data_source":{"id":"17658"},"glyph":{"id":"17659"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17660"},"selection_glyph":null,"view":{"id":"17662"}},"id":"17661","type":"GlyphRenderer"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.663971371122512},"y":{"value":0}},"id":"17645","type":"Circle"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.74129666559688},"y":{"value":1.95}},"id":"17719","type":"Circle"},{"attributes":{"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17664","type":"Line"},{"attributes":{"data":{"x":[2.393576367885318,6.880909766712211],"y":[1.95,1.95]},"selected":{"id":"17798"},"selection_policy":{"id":"17797"}},"id":"17713","type":"ColumnDataSource"},{"attributes":{},"id":"17787","type":"UnionRenderers"},{"attributes":{},"id":"17619","type":"UndoTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.59253463805363},"y":{"value":0.6}},"id":"17675","type":"Circle"},{"attributes":{"source":{"id":"17713"}},"id":"17717","type":"CDSView"},{"attributes":{"overlay":{"id":"17623"}},"id":"17618","type":"LassoSelectTool"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17669","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17715","type":"Line"},{"attributes":{},"id":"17620","type":"SaveTool"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17724","type":"Line"},{"attributes":{},"id":"17788","type":"Selection"},{"attributes":{"data_source":{"id":"17718"},"glyph":{"id":"17719"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17720"},"selection_glyph":null,"view":{"id":"17722"}},"id":"17721","type":"GlyphRenderer"},{"attributes":{},"id":"17617","type":"WheelZoomTool"},{"attributes":{},"id":"17803","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"17728"},"glyph":{"id":"17729"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17730"},"selection_glyph":null,"view":{"id":"17732"}},"id":"17731","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17635","type":"Line"},{"attributes":{},"id":"17615","type":"PanTool"},{"attributes":{},"id":"17614","type":"ResetTool"},{"attributes":{"data":{},"selected":{"id":"17800"},"selection_policy":{"id":"17799"}},"id":"17718","type":"ColumnDataSource"},{"attributes":{},"id":"17804","type":"Selection"},{"attributes":{"data_source":{"id":"17643"},"glyph":{"id":"17644"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17645"},"selection_glyph":null,"view":{"id":"17647"}},"id":"17646","type":"GlyphRenderer"},{"attributes":{"data":{"x":[-0.7842313478998125,9.985046696913068],"y":[2.55,2.55]},"selected":{"id":"17808"},"selection_policy":{"id":"17807"}},"id":"17738","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"17622"}},"id":"17616","type":"BoxZoomTool"},{"attributes":{"source":{"id":"17718"}},"id":"17722","type":"CDSView"},{"attributes":{"axis":{"id":"17610"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"17613","type":"Grid"},{"attributes":{"data_source":{"id":"17738"},"glyph":{"id":"17739"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17740"},"selection_glyph":null,"view":{"id":"17742"}},"id":"17741","type":"GlyphRenderer"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17665","type":"Line"},{"attributes":{"data":{},"selected":{"id":"17776"},"selection_policy":{"id":"17775"}},"id":"17658","type":"ColumnDataSource"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17729","type":"Line"},{"attributes":{},"id":"17789","type":"UnionRenderers"},{"attributes":{"source":{"id":"17723"}},"id":"17727","type":"CDSView"},{"attributes":{},"id":"17790","type":"Selection"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17639","type":"Line"},{"attributes":{"source":{"id":"17653"}},"id":"17657","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17725","type":"Line"},{"attributes":{},"id":"17604","type":"LinearScale"},{"attributes":{},"id":"17805","type":"UnionRenderers"},{"attributes":{"data":{},"selected":{"id":"17770"},"selection_policy":{"id":"17769"}},"id":"17643","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"17735","type":"Circle"},{"attributes":{},"id":"17607","type":"BasicTicker"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.181352887007439},"y":{"value":2.25}},"id":"17734","type":"Circle"},{"attributes":{},"id":"17806","type":"Selection"},{"attributes":{"toolbars":[{"id":"17624"}],"tools":[{"id":"17614"},{"id":"17615"},{"id":"17616"},{"id":"17617"},{"id":"17618"},{"id":"17619"},{"id":"17620"},{"id":"17621"}]},"id":"17815","type":"ProxyToolbar"},{"attributes":{"data":{"x":[1.9127415870254314,7.0411499662839026],"y":[2.25,2.25]},"selected":{"id":"17804"},"selection_policy":{"id":"17803"}},"id":"17728","type":"ColumnDataSource"},{"attributes":{"source":{"id":"17728"}},"id":"17732","type":"CDSView"},{"attributes":{"source":{"id":"17658"}},"id":"17662","type":"CDSView"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17730","type":"Line"},{"attributes":{"formatter":{"id":"17761"},"ticker":{"id":"17607"}},"id":"17606","type":"LinearAxis"},{"attributes":{"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17739","type":"Line"},{"attributes":{"data_source":{"id":"17733"},"glyph":{"id":"17734"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17735"},"selection_glyph":null,"view":{"id":"17737"}},"id":"17736","type":"GlyphRenderer"},{"attributes":{},"id":"17791","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"17743"},"glyph":{"id":"17744"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17745"},"selection_glyph":null,"view":{"id":"17747"}},"id":"17746","type":"GlyphRenderer"},{"attributes":{"data":{},"selected":{"id":"17806"},"selection_policy":{"id":"17805"}},"id":"17733","type":"ColumnDataSource"},{"attributes":{},"id":"17792","type":"Selection"},{"attributes":{"data":{"x":[2.2921199846409115,6.479187946875487],"y":[0.8999999999999999,0.8999999999999999]},"selected":{"id":"17786"},"selection_policy":{"id":"17785"}},"id":"17683","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#fa7c17"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"17660","type":"Circle"},{"attributes":{"source":{"id":"17733"}},"id":"17737","type":"CDSView"},{"attributes":{},"id":"17807","type":"UnionRenderers"},{"attributes":{},"id":"17808","type":"Selection"},{"attributes":{"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17744","type":"Line"},{"attributes":{},"id":"17762","type":"BasicTickFormatter"},{"attributes":{"source":{"id":"17738"}},"id":"17742","type":"CDSView"},{"attributes":{"callback":null},"id":"17621","type":"HoverTool"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.409527442378456},"y":{"value":0.3}},"id":"17659","type":"Circle"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17740","type":"Line"},{"attributes":{},"id":"17793","type":"UnionRenderers"},{"attributes":{"data":{"x":[2.1521098502407368,6.9643589964055215],"y":[0.6,0.6]},"selected":{"id":"17780"},"selection_policy":{"id":"17779"}},"id":"17668","type":"ColumnDataSource"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#2a2eec"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"17750","type":"Circle"},{"attributes":{"fill_color":{"value":"#2a2eec"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":3.4162688471038556},"y":{"value":2.55}},"id":"17749","type":"Circle"},{"attributes":{},"id":"17794","type":"Selection"},{"attributes":{"data_source":{"id":"17673"},"glyph":{"id":"17674"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17675"},"selection_glyph":null,"view":{"id":"17677"}},"id":"17676","type":"GlyphRenderer"},{"attributes":{"data":{"x":[1.9790083397469873,5.455951625836456],"y":[2.55,2.55]},"selected":{"id":"17810"},"selection_policy":{"id":"17809"}},"id":"17743","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"17683"},"glyph":{"id":"17684"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17685"},"selection_glyph":null,"view":{"id":"17687"}},"id":"17686","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17743"}},"id":"17747","type":"CDSView"},{"attributes":{},"id":"17809","type":"UnionRenderers"},{"attributes":{"data":{},"selected":{"id":"17782"},"selection_policy":{"id":"17781"}},"id":"17673","type":"ColumnDataSource"},{"attributes":{"line_alpha":0.1,"line_color":"#2a2eec","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17745","type":"Line"},{"attributes":{},"id":"17811","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"17693"},"glyph":{"id":"17694"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17695"},"selection_glyph":null,"view":{"id":"17697"}},"id":"17696","type":"GlyphRenderer"},{"attributes":{"formatter":{"id":"17762"},"major_label_overrides":{"0.44999999999999996":"Non Centered: mu","2.0999999999999996":"Centered: mu"},"ticker":{"id":"17757"}},"id":"17610","type":"LinearAxis"},{"attributes":{},"id":"17810","type":"Selection"},{"attributes":{"data_source":{"id":"17748"},"glyph":{"id":"17749"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"17750"},"selection_glyph":null,"view":{"id":"17752"}},"id":"17751","type":"GlyphRenderer"},{"attributes":{"source":{"id":"17673"}},"id":"17677","type":"CDSView"},{"attributes":{"axis":{"id":"17606"},"ticker":null},"id":"17609","type":"Grid"},{"attributes":{"toolbar":{"id":"17815"},"toolbar_location":"above"},"id":"17816","type":"ToolbarBox"},{"attributes":{"bounds":"auto","min_interval":1},"id":"17755","type":"DataRange1d"},{"attributes":{"data":{},"selected":{"id":"17812"},"selection_policy":{"id":"17811"}},"id":"17748","type":"ColumnDataSource"},{"attributes":{"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17684","type":"Line"},{"attributes":{"source":{"id":"17678"}},"id":"17682","type":"CDSView"},{"attributes":{"source":{"id":"17748"}},"id":"17752","type":"CDSView"},{"attributes":{},"id":"17795","type":"UnionRenderers"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":1.5,"x":{"field":"x"},"y":{"field":"y"}},"id":"17680","type":"Line"},{"attributes":{"line_alpha":0.1,"line_color":"#fa7c17","line_width":3.0,"x":{"field":"x"},"y":{"field":"y"}},"id":"17685","type":"Line"},{"attributes":{"fill_color":{"value":"#fa7c17"},"line_color":{"value":"#1f77b4"},"size":{"units":"screen","value":4.5},"x":{"value":4.3706630373472235},"y":{"value":0.8999999999999999}},"id":"17689","type":"Circle"},{"attributes":{"source":{"id":"17683"}},"id":"17687","type":"CDSView"}],"root_ids":["17817"]},"title":"Bokeh Application","version":"2.2.3"}}';
                  var render_items = [{"docid":"4554dd45-ede1-4a8b-ae8c-50592a0bfdbb","root_ids":["17817"],"roots":{"17817":"a65b432a-9cd3-4ac6-bf42-23869875eb56"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();